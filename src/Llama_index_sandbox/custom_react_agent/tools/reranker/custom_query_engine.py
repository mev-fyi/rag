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
            'remainder': 0.9
        },
        'youtube_weights': {
            'Bell Curve 2023': 0.95,
            'Fenbushi Capital': 0.9,
            'SevenX Ventures': 0.9,
            'Tim Roughgarden Lectures': 0.9,
            'remainder': 0.8,
        },
    }
    authors_list = {
        'RIG': [],
        'Flashbots': [],
        'EF': [],
    }
    authors_weights = {
        'RIG': 1.05,
        'Flashbots': 1.05,
        'EF': 1.05,
    }

    def nodes_reranker(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        for node_with_score in nodes_with_score:
            score = node_with_score.score
            document_type = node_with_score.node.metadata.get('document_type', 'UNSPECIFIED')
            authors = node_with_score.node.metadata.get('authors', 'UNSPECIFIED AUTHORS')
            link = node_with_score.node.metadata.get('video_link', 'UNSPECIFIED LINK') \
                if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value \
                else node_with_score.node.metadata.get('pdf_link', 'UNSPECIFIED LINK')

            # Document type weight adjustments
            weight_key = document_type.lower() + '_weights'
            if weight_key in self.document_weights:
                for source, weight in self.document_weights[weight_key].items():
                    if source in link or source == 'remainder':
                        score *= weight
                        break  # Break the loop once a match is found

            # Author weight adjustment
            for author in authors.split(','):
                author = author.strip()
                for firm, authors_in_firm in self.authors_list.items():
                    if author in authors_in_firm:
                        score *= self.authors_weights.get(firm, 1)

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
