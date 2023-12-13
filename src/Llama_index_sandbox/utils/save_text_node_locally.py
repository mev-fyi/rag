import ray
import pyarrow as pa
import pandas as pd
from typing import List
from llama_index.schema import TextNode  # Assuming TextNode class is defined here

# Start Ray.
ray.init()
# TODO 2023-11-04: finish implementation of local embedding saving to enable sweeping across parameters e.g. distance metric


def save_text_node_chunk(file_name, text_nodes_chunk):
    df = pd.DataFrame([text_node.__dict__ for text_node in text_nodes_chunk])
    table = pa.Table.from_pandas(df)
    pa.parquet.write_table(table, file_name)
    return file_name


def load_text_node_chunk(file_name):
    table = pa.parquet.read_table(file_name)
    df = table.to_pandas()
    return [TextNode(**row) for index, row in df.iterrows()]


def save_text_nodes_at_scale(text_nodes: List[TextNode], chunk_size=100):
    # Divide the text_nodes list into chunks.
    chunks = [text_nodes[i:i + chunk_size] for i in range(0, len(text_nodes), chunk_size)]

    # Save each chunk to a separate Parquet file in parallel using `map`.
    # file_names = [f'text_nodes_chunk_{i}.parquet' for i in range(len(chunks)]
    # ray.get(ray.experimental.Batches(chunks).for_each(lambda chunk, i: save_text_node_chunk(file_names[i], chunk)))

    # print(f"Saved the following Parquet files: {file_names}")


def load_text_nodes_at_scale(parquet_files: List[str]):
    # Load each Parquet file in parallel using `map`.
    loaded_chunks = ray.experimental.Batches(parquet_files).for_each(load_text_node_chunk)

    # Flatten the loaded chunks to obtain the final list of TextNode objects.
    text_nodes = [text_node for chunk in loaded_chunks for text_node in chunk]

    return text_nodes