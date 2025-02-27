from datetime import datetime
from itertools import product
import os

import src.Llama_index_sandbox.utils.utils
from src.Llama_index_sandbox import embed
import src.Llama_index_sandbox.evaluation.evaluation_constants as config
from src.Llama_index_sandbox.utils.utils import get_last_index_embedding_params, root_directory

root_dir = root_directory()
index_dir = f"{root_dir}/.storage/research_pdf/"


class Config:
    def __init__(self):
        # Ingestion Parameters
        self.engine = 'chat'
        self.query_engine_as_tool = True
        self.reset_chat = os.environ.get('RESET_CHAT')
        self.add_new_transcripts = False
        self.stream = True
        self.num_files = None  # 10

        # Indexing Parameters
        self.VECTOR_SPACE_DISTANCE_METRIC = ['cosine']
        self.NUM_CHUNKS_RETRIEVED = [int(os.environ.get('NUM_CHUNKS_RETRIEVED'))]  # config.NUM_CHUNKS_RETRIEVED
        self.NUM_CHUNKS_SEARCHED_FOR_RERANKING = [int(os.environ.get('NUM_CHUNKS_SEARCHED_FOR_RERANKING'))]  # config.NUM_CHUNKS_RETRIEVED
        self.CHUNK_SIZES = [750]  # config.CHUNK_SIZES
        self.CHUNK_OVERLAPS = [10]  # config.CHUNK_OVERLAPS
        self.EMBEDDING_MODELS = [os.environ.get('EMBEDDING_MODEL_NAME')]  # ["BAAI/bge-large-en-v1.5"]  # config.EMBEDDING_MODELS
        self.INFERENCE_MODELS = [os.environ.get('LLM_MODEL_NAME')]  # config.INFERENCE_MODELS
        self.EMBEDDING_DIMENSIONS = {
            "thenlper/gte-base": 768,
            "thenlper/gte-large": 1024,
            "BAAI/bge-large-en": 1024,
            "BAAI/bge-large-en-v1.5": 1024,
            "text-embedding-ada-002": 1536,
        }
        self.MAX_CONTEXT_LENGTHS = {
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "meta-llama/Llama-2-7b-chat-hf": 4096,
            "meta-llama/Llama-2-13b-chat-hf": 4096,
            "meta-llama/Llama-2-70b-chat-hf": 4096,
        }

    def get_index_output_dir(self):
        embedding_model_name = self.EMBEDDING_MODELS[0]
        text_splitter_chunk_size = self.CHUNK_SIZES[0]
        text_splitter_chunk_overlap_percentage = self.CHUNK_OVERLAPS[0]

        # Format the filename
        if '/' in embedding_model_name:
            embedding_model_name = embedding_model_name.split('/')[-1]
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
        name = f"{date_str}_{embedding_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}"
        output_dir = index_dir + name
        # check if index_dir and if not create it
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        return output_dir

    def get_index_params(self, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model, llm_model_name, vector_space_distance_metric):
        index_embedding_model_name, index_text_splitter_chunk_size, index_chunk_overlap, index_vector_space_distance_metric = get_last_index_embedding_params()
        recreate_index = (
            text_splitter_chunk_size != index_text_splitter_chunk_size or
            text_splitter_chunk_overlap_percentage != index_chunk_overlap or
            embedding_model_name.split('/')[-1] != index_embedding_model_name or
            vector_space_distance_metric != index_vector_space_distance_metric  # TODO 2023-11-02: implement vector_space_distance_metric
        )

        index_params = {
            "recreate_index": recreate_index,
            "text_splitter_chunk_size": text_splitter_chunk_size,
            "text_splitter_chunk_overlap_percentage": text_splitter_chunk_overlap_percentage,
            "embedding_model_name": embedding_model_name,
            "embedding_model": embedding_model,
            "llm_model_name": llm_model_name,
            'vector_space_distance_metric': vector_space_distance_metric,
            "add_new_transcripts": self.add_new_transcripts,
            "num_files": self.num_files
        }
        return index_params

    def get_full_combinations(self):
        index_combinations = product(self.CHUNK_SIZES, self.CHUNK_OVERLAPS, self.EMBEDDING_MODELS, [src.Llama_index_sandbox.utils.utils.get_embedding_model(embedding_model_name=e) for e in self.EMBEDDING_MODELS], self.INFERENCE_MODELS, self.VECTOR_SPACE_DISTANCE_METRIC)
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

