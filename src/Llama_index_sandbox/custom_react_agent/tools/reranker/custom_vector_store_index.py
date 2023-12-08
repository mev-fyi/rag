from abc import ABC
from typing import Any

from llama_index import VectorStoreIndex
from llama_index.indices.query.base import BaseQueryEngine

from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine


class CustomVectorStoreIndex(VectorStoreIndex):

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        retriever = self.as_retriever(**kwargs)

        kwargs["retriever"] = retriever
        if "service_context" not in kwargs:
            kwargs["service_context"] = self._service_context
        return CustomQueryEngine.from_args(**kwargs)

    @classmethod
    def from_vector_store_index(cls, vector_store_index):
        # Create a new instance of CustomVectorStoreIndex using the attributes of vector_store_index
        return cls(
            nodes=getattr(vector_store_index, '_nodes', None),
            index_struct=getattr(vector_store_index, '_index_struct', None),
            service_context=getattr(vector_store_index, '_service_context', None),
            storage_context=getattr(vector_store_index, '_storage_context', None),
            use_async=getattr(vector_store_index, '_use_async', False),
            store_nodes_override=getattr(vector_store_index, '_store_nodes_override', False),
            show_progress=getattr(vector_store_index, '_show_progress', False),
            # Include other relevant attributes if there are any
        )