from llama_index import QueryBundle
from llama_index.callbacks import EventPayload, CBEventType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import NodeWithScore
from typing_extensions import List

from src.Llama_index_sandbox.constants import DOCUMENT_TYPES


class CustomQueryEngine(RetrieverQueryEngine):
    document_weights = {
        'article_weights': {
            'ethresear.ch': 1,
            'writings.flashbots': 1,
            'frontier.tech': 0.95,
            'research.anoma': 0.95,
            'default': 0.9
        },
        'youtube_weights': {
            'Bell Curve 2023': 0.95,
            'Fenbushi Capital': 0.9,
            'SevenX Ventures': 0.9,
            'Tim Roughgarden Lectures': 0.9,
            'default': 0.8,
        },
    }
    authors_list = {
        'EF': [
            'https://ethresear.ch/u/mikeneuder',
            ''
        ],
        'Flashbots': [
            'https://collective.flashbots.net/u/Quintus',
            'https://collective.flashbots.net/u/flashbots',
            'https://collective.flashbots.net/u/bert',
            'https://collective.flashbots.net/u/chayoterabit',
            'https://collective.flashbots.net/u/dmarz',
            'https://collective.flashbots.net/u/bert',
            'https://collective.flashbots.net/u/sarah',
            'https://collective.flashbots.net/u/elainehu/',
            'https://collective.flashbots.net/u/Fred',
            'https://collective.flashbots.net/u/system/summary',
        ],
    }
    authors_weights = {
        'Flashbots': 1.05,
        'EF': 1.05,
    }

    def nodes_reranker(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        for node_with_score in nodes_with_score:
            score = node_with_score.score
            document_type = node_with_score.node.metadata.get('document_type', 'UNSPECIFIED')
            authors = node_with_score.node.metadata.get('channel_name', 'UNSPECIFIED LINK') \
                if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value \
                else node_with_score.node.metadata.get('authors', 'UNSPECIFIED LINK')
            link = node_with_score.node.metadata.get('video_link', 'UNSPECIFIED LINK') \
                if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value \
                else node_with_score.node.metadata.get('pdf_link', 'UNSPECIFIED LINK')

            # Document type weight adjustments
            weight_key = document_type.lower() + '_weights'
            if weight_key in self.document_weights:
                matched = False
                for source, weight in self.document_weights[weight_key].items():
                    if source in link:
                        score *= weight
                        matched = True
                        break  # Break the loop once a match is found
                if not matched and 'default' in self.document_weights[weight_key]:
                    score *= self.document_weights[weight_key]['default']

            # Author weight adjustment
            author_matched = False
            for author in authors.split(','):
                author = author.strip()
                for firm, authors_in_firm in self.authors_list.items():
                    if author in authors_in_firm:
                        score *= self.authors_weights.get(firm, 1)
                        author_matched = True
                        break  # Break the inner loop if an author matches
                if author_matched:
                    break  # Break the outer loop if any author matches

            if not author_matched and 'default' in self.authors_weights:
                score *= self.authors_weights['default']  # Apply default weight for authors if no author matched

            node_with_score.score = score
        return nodes_with_score



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
