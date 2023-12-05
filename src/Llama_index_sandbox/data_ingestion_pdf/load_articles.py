import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union
import pandas as pd
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.utils import timeit


@timeit
def load_single_pdf(paper_details_df, file_path, loader=PyMuPDFReader()):
    try:
        documents = loader.load(file_path=file_path)
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        # Find the corresponding row in the DataFrame
    title = os.path.basename(file_path).replace('.pdf', '')
    paper_row = paper_details_df[paper_details_df['title'] == title]

    if not paper_row.empty:
        # Update metadata
        for document in documents:
            document.metadata.update({
                'document_type': DOCUMENT_TYPES.ARTICLE.value,
                'title': paper_row.iloc[0]['title'],
                'pdf_link': paper_row.iloc[0]['article'],
            })

    return documents

@timeit
def load_pdfs(directory_path: Union[str, Path], num_files: int = None):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    research_papers_path = f"{root_dir}/datasets/evaluation_data/articles_updated.csv"
    paper_details_df = pd.read_csv(research_papers_path)
    partial_load_single_pdf = partial(load_single_pdf, paper_details_df=paper_details_df)

    pdf_loaded_count = 0
    files_gen = directory_path.glob("*.pdf")

    if num_files is not None:
        files = [next(files_gen) for _ in range(num_files) if files_gen]
    else:
        files = list(files_gen)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(partial_load_single_pdf, file_path=pdf_file): pdf_file for pdf_file in files}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                pdf_loaded_count += 1
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}, removing file: {e}")
                os.remove(pdf_file)

    logging.info(f"Successfully loaded [{pdf_loaded_count}] documents from PDF files.")
    return all_documents


# Example usage
pdf_directory = f"{root_dir}/datasets/evaluation_data/articles_2023-12-05"
load_pdfs(pdf_directory)
