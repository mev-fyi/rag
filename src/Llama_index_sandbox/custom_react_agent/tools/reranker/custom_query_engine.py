import logging
import pickle
from itertools import product

import tldextract
from llama_index import QueryBundle
from llama_index.callbacks import EventPayload, CBEventType
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import NodeWithScore
from typing import List
import os
import heapq
from urllib.parse import urlparse
from tldextract import extract  # You might need to install this package

from src.Llama_index_sandbox.constants import DOCUMENT_TYPES
from src.anyscale_sandbox.utils import root_directory


class CustomQueryEngine(RetrieverQueryEngine):

    document_weights = {
        f'{DOCUMENT_TYPES.ARTICLE.value}_weights': {
            'ethresear.ch': 1,
            'writings.flashbots.net': 1,
            'frontier.tech': 0.95,
            'research.anoma.net': 0.95,
            'default': 0.9
        },
        f'{DOCUMENT_TYPES.YOUTUBE_VIDEO.value}_weights': {
            'Flashbots': 1,
            'Bell Curve 2023': 0.95,
            'Fenbushi Capital': 0.9,
            'SevenX Ventures': 0.9,
            'Research Day': 0.9,
            'Tim Roughgarden Lectures': 0.9,
            'default': 0.8,
        },
        f'{DOCUMENT_TYPES.RESEARCH_PAPER.value}_weights': {
            'default': 1
        },
        f'unspecified_weights': {  # default case for absent metadata
            'default': 0.7
        }
    }
    authors_list = {
        'EF': {
            'https://ethresear.ch/u/mikeneuder',
            'https://ethresear.ch/u/barnabe',
        },
        'Flashbots': {
            'https://collective.flashbots.net/u/Quintus',
            'https://collective.flashbots.net/u/flashbots',
            'https://collective.flashbots.net/u/chayoterabit',
            'https://collective.flashbots.net/u/dmarz',
            'https://collective.flashbots.net/u/bert',
            'https://collective.flashbots.net/u/sarah',
            'https://collective.flashbots.net/u/elainehu/',
            'https://collective.flashbots.net/u/Fred',
            'https://collective.flashbots.net/u/system/summary',
        },
    }
    authors_weights = {
        'Flashbots': 1.05,
        'EF': 1.05,
        'default': 1,
    }

    edge_case_of_content_always_cited = ['Editorial content: Strategies and tactics | Sonal Chokshi']

    edge_case_set = set(edge_case_of_content_always_cited)

    # Pre-compute mappings for document weights
    document_weight_mappings = {
        key: {source: weight for source, weight in weights.items() if source != 'default'}
        for key, weights in document_weights.items()
    }
    default_weights = {
        key: weights.get('default', 1) for key, weights in document_weights.items()
    }

    # Pre-compute an author-to-weight mapping
    author_weight_mapping = {}
    for firm, authors in authors_list.items():
        weight = authors_weights.get(firm, 1)
        for author in authors:
            author_weight_mapping[author] = weight

    # Add a default weight
    author_weight_mapping['default'] = authors_weights.get('default', 1)

    # File to store the precomputed effective weights
    weights_file = f"{root_directory()}/datasets/evaluation_data/effective_weights.pkl"

    @staticmethod
    def load_or_compute_weights(document_weight_mappings, weights_file, authors_list, authors_weights, recompute_weights=False):
        def precompute_effective_weights(document_weight_mappings, authors_weights, authors_list):
            effective_weights = {}

            # Generate all possible combinations for articles with triplets
            for group, authors in authors_list.items():
                group_weight = authors_weights.get(group, 1)
                for author_url in authors:
                    author_extracted = tldextract.extract(author_url)
                    author_domain = author_extracted.domain + '.' + author_extracted.suffix

                    for source, weight in document_weight_mappings.get(DOCUMENT_TYPES.ARTICLE.value + '_weights', {}).items():
                        source_extracted = tldextract.extract(source)
                        source_domain = source_extracted.domain + '.' + source_extracted.suffix

                        # Match domains for articles and calculate weight
                        if author_domain == source_domain:
                            key = (DOCUMENT_TYPES.ARTICLE.value, source, author_url)
                            effective_weights[key] = weight * group_weight

            # Generate pairs for research papers and videos
            for document_type, document_weights in document_weight_mappings.items():
                if document_type != DOCUMENT_TYPES.ARTICLE.value + '_weights':
                    for source, weight in document_weights.items():
                        key = (document_type, source)
                        effective_weights[key] = document_weights.get(source, document_weights.get('default', 1))

            return effective_weights

        # Check if the file exists and the size of data structures hasn't changed
        if os.path.exists(weights_file) and (not recompute_weights):
            with open(weights_file, 'rb') as f:
                return pickle.load(f)
        else:
            effective_weights = precompute_effective_weights(document_weight_mappings=document_weight_mappings, authors_list=authors_list, authors_weights=authors_weights)
            with open(weights_file, 'wb') as f:
                pickle.dump(effective_weights, f)
            return effective_weights

    # Load or compute the effective weights
    effective_weights = load_or_compute_weights(document_weight_mappings=document_weight_mappings, weights_file=weights_file, authors_list=authors_list, authors_weights=authors_weights)

    def nodes_reranker(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        NUM_CHUNKS_RETRIEVED = int(os.environ.get('NUM_CHUNKS_RETRIEVED', '10'))

        for node_with_score in nodes_with_score:
            score = node_with_score.score
            document_type = node_with_score.node.metadata.get('document_type', 'UNSPECIFIED').lower()
            document_name = node_with_score.node.metadata.get('title', 'UNSPECIFIED')

            if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                source = node_with_score.node.metadata.get('channel_name', 'UNSPECIFIED LINK').strip()
                weight_key = (document_type + '_weights', source)
                effective_weight = self.effective_weights.get(
                    weight_key,
                    self.document_weights[document_type + '_weights'].get('default', 1)
                )
            else:
                link = node_with_score.node.metadata.get('pdf_link', 'UNSPECIFIED LINK').strip()
                extracted = tldextract.extract(link)
                domain = f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}".strip('.')
                author_url = node_with_score.node.metadata.get('authors', 'UNSPECIFIED LINK').strip()
                weight_key = (document_type, domain, author_url)  #  + '_weights'
                effective_weight = self.effective_weights.get(
                    weight_key,
                    self.document_weights[document_type + '_weights'].get(domain, self.document_weights[document_type + '_weights'].get('default', 1))
                )

            score *= effective_weight

            # Special case adjustment
            if document_name in self.edge_case_set:
                score *= 0.8

            node_with_score.score = score

        # Optional logging if in local environment
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(nodes_with_score[:NUM_CHUNKS_RETRIEVED], f"Top {NUM_CHUNKS_RETRIEVED} nodes before rerank")

        # Get the top NUM_CHUNKS_RETRIEVED nodes based on score
        top_nodes = heapq.nlargest(NUM_CHUNKS_RETRIEVED, nodes_with_score, key=lambda x: x.score)

        # Optional logging if in local environment
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(top_nodes, f"Re-ranked top {NUM_CHUNKS_RETRIEVED} nodes")

        return top_nodes

    def nodes_reranker_manual(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Reranks a list of nodes based on their scores, adjusting these scores according to predefined weights.
        The function considers both document type and author influence in the scoring system.

        - For YouTube videos, the score is adjusted based on the authors' weights as defined in 'authors_weights'.
        - For articles and other document types, the score is adjusted based on the link's match with 'document_weights'.
        - If no specific match is found, a default weight is applied.

        Args:
        nodes_with_score (List[NodeWithScore]): A list of NodeWithScore objects, each containing a node and its associated score.

        Returns:
        List[NodeWithScore]: The same list of nodes, but reordered based on their adjusted scores, returning only the top results as defined by the 'NUM_CHUNKS_RETRIEVED' environment variable.
        """
        NUM_CHUNKS_RETRIEVED = int(os.environ.get('NUM_CHUNKS_RETRIEVED'))

        for node_with_score in nodes_with_score:
            score = node_with_score.score
            document_type = node_with_score.node.metadata.get('document_type', 'UNSPECIFIED')
            document_name = node_with_score.node.metadata.get('title', 'UNSPECIFIED')
            authors = node_with_score.node.metadata.get('channel_name', 'UNSPECIFIED LINK') \
                if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value \
                else node_with_score.node.metadata.get('authors', 'UNSPECIFIED LINK')
            link = node_with_score.node.metadata.get('video_link', 'UNSPECIFIED LINK') \
                if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value \
                else node_with_score.node.metadata.get('pdf_link', 'UNSPECIFIED LINK')

            authors = authors.strip()
            link = link.strip()

            score = self.adjust_score_for_document_type(document_type, authors, link, score, document_name)
            score = self.adjust_score_for_authors(authors, score, document_type)
            node_with_score.score = score

        # reorder the node_with_score objects within the list based on the score
        # Log unique file names from the top k results of the sorted list
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(nodes_with_score[:NUM_CHUNKS_RETRIEVED], f"Initial top {NUM_CHUNKS_RETRIEVED} nodes before rerank")

        nodes_with_score = heapq.nlargest(NUM_CHUNKS_RETRIEVED, nodes_with_score, key=lambda x: x.score)

        # Log unique file names from the top k results of the truncated list
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(nodes_with_score, f"Re-ranked top {NUM_CHUNKS_RETRIEVED} nodes")

        # TODO 2023-12-10: if the next node is in the same document, should we still include it or not?
        return nodes_with_score

    def adjust_score_for_document_type(self, document_type, authors, link, score, document_name):
        weight_key = document_type.lower() + '_weights'
        if weight_key in self.document_weights:
            source = authors if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value else link
            weight = self.find_weight_for_source(source, weight_key)
            score *= weight

            # Apply special case adjustment if applicable
            if document_name in self.edge_case_set:
                score *= 0.8

        return score

    def find_weight_for_source(self, source, weight_key):
        extracted = extract(source)

        # Join the subdomain and domain to get the key
        # e.g., 'blog.example.co.uk' will become 'blog.example'
        domain_key = '.'.join(filter(None, [extracted.subdomain, extracted.domain]))

        # Check if the domain_key is directly in the document_weight_mappings
        weight = self.document_weight_mappings.get(weight_key, {}).get(domain_key, self.default_weights.get(weight_key, 1))
        return weight

    def adjust_score_for_authors(self, authors, score, document_type):
        if document_type != DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
            author_matched = False
            for author in authors.split(','):
                author = author.strip()
                if author in self.author_weight_mapping:
                    score *= self.author_weight_mapping[author]
                    author_matched = True
                    break  # Break the loop if an author matches

            if not author_matched:
                score *= self.author_weight_mapping['default']  # Apply default weight if no author matched

        return score

    def log_unique_filenames(self, nodes: List[NodeWithScore], context: str):
        unique_files_info = {}
        document_type_count = {}

        logging.info(f"Logging unique file names and document types in context: {context}")

        # Count occurrences of each title
        title_count = {}
        for node in nodes:
            title = node.node.metadata.get('title', 'UNSPECIFIED FILE')
            title_count[title] = title_count.get(title, 0) + 1

        for node in nodes:
            filename = node.node.metadata.get('title', 'UNSPECIFIED FILE')
            document_type = node.node.metadata.get('document_type', 'UNSPECIFIED')

            file_key = (document_type, filename)
            if file_key not in unique_files_info:
                unique_files_info[file_key] = {'chunk_count': title_count[filename], 'index': len(unique_files_info) + 1}

            document_type_count[document_type] = document_type_count.get(document_type, 0) + 1

        for file_key, info in unique_files_info.items():
            document_type, filename = file_key
            logging.info(f"Unique file #{info['index']}: Document Type: [{document_type}], Filename: [{filename}], Chunks Retrieved: [{info['chunk_count']}]")

        # Constructing a single string for all document type counts
        document_type_counts_str = ", ".join(f"{doc_type}: {count}" for doc_type, count in document_type_count.items())
        logging.info(f"Document Type chunk-distribution: {document_type_counts_str}")

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)

                nodes = self.nodes_reranker(nodes_with_score=nodes)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
