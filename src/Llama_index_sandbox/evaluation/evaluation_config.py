from itertools import product

from src.Llama_index_sandbox import embed
from src.Llama_index_sandbox.evaluation.evaluation_constants import NUM_CHUNKS_RETRIEVED, CHUNK_SIZES, CHUNK_OVERLAPS, EMBEDDING_MODELS, INFERENCE_MODELS
from src.Llama_index_sandbox.utils import get_last_index_embedding_params


class Config:
    def __init__(self):
        # Ingestion Parameters
        self.engine = 'chat'
        self.query_engine_as_tool = True
        self.reset_chat = True
        self.add_new_transcripts = False
        self.stream = True
        self.num_files = None

        # Indexing Parameters
        self.NUM_CHUNKS_RETRIEVED = NUM_CHUNKS_RETRIEVED
        self.CHUNK_SIZES = CHUNK_SIZES
        self.CHUNK_OVERLAPS = CHUNK_OVERLAPS
        self.EMBEDDING_MODELS = EMBEDDING_MODELS
        self.INFERENCE_MODELS = INFERENCE_MODELS

    def get_index_params(self, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model):
        index_embedding_model_name, index_text_splitter_chunk_size, index_chunk_overlap = get_last_index_embedding_params()
        recreate_index = (
            text_splitter_chunk_size != index_text_splitter_chunk_size or
            text_splitter_chunk_overlap_percentage != index_chunk_overlap or
            embedding_model_name.split('/')[-1] != index_embedding_model_name
        )

        index_params = {
            "recreate_index": recreate_index,
            "text_splitter_chunk_size": text_splitter_chunk_size,
            "text_splitter_chunk_overlap_percentage": text_splitter_chunk_overlap_percentage,
            "embedding_model_name": embedding_model_name,
            "embedding_model": embedding_model,
            "add_new_transcripts": self.add_new_transcripts,
            "num_files": self.num_files
        }
        return index_params

    def get_full_combinations(self):
        index_combinations = product(self.CHUNK_SIZES, self.CHUNK_OVERLAPS, self.EMBEDDING_MODELS, [embed.get_embedding_model(embedding_model_name=e) for e in self.EMBEDDING_MODELS])
        return index_combinations

    def get_inference_params(self, llm_model_name, similarity_top_k, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model):
        full_combination = {
            "similarity_top_k": similarity_top_k,
            "text_splitter_chunk_size": text_splitter_chunk_size,
            "text_splitter_chunk_overlap_percentage": text_splitter_chunk_overlap_percentage,
            "embedding_model_name": embedding_model_name,
            "embedding_model": embedding_model,
            "llm_model_name": llm_model_name,
            "add_new_transcripts": self.add_new_transcripts,
            "num_files": self.num_files,
            "stream": self.stream
        }
        return full_combination

