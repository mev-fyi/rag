import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from src.Llama_index_sandbox.custom_pymupdfreader.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.utils.utils import timeit, save_successful_load_to_csv, compute_new_entries


def load_single_pdf(paper_details_df, file_path, current_df, loader=PyMuPDFReader()):
    try:
        documents = loader.load(file_path=file_path)
        title = os.path.basename(file_path).replace('.pdf', '').replace('<slash>', '/')

        is_in_current_df = True if not current_df[current_df['title'] == title].empty else False
        if is_in_current_df:
            return []

        paper_row = paper_details_df[paper_details_df['Title'] == title]  # if there is a match it means that we need to add that title for which we have metadata

        if not paper_row.empty:
            assert paper_row.iloc[0]['Title'] != np.nan, f"Title is NaN for {paper_row.iloc[0]['Link']}"
            assert paper_row.iloc[0]['Author'] != np.nan, f"Author is NaN for {paper_row.iloc[0]['Link']}"
            assert paper_row.iloc[0]['Link'] != np.nan, f"Link is NaN for {paper_row.iloc[0]['Link']}"
            assert paper_row.iloc[0]['Release Date'] != np.nan, f"Release Date is NaN for {paper_row.iloc[0]['Link']}"

            # Update metadata
            for document in documents:
                if 'file_path' in document.metadata.keys():
                    del document.metadata['file_path']

                if not document.text:
                    logging.error(f"Found empty document text in [{title}] with [{file_path}]")
                    continue
                document.text.replace('', '')

                document.metadata.update({
                    'document_type': DOCUMENT_TYPES.ARTICLE.value,
                    'title': title,
                    'authors': str(paper_row.iloc[0]['Author']),
                    'pdf_link': str(paper_row.iloc[0]['Link']),
                    'release_date': str(paper_row.iloc[0]['Release Date']),
                })

        else:
            # Update metadata
            for document in documents:
                if 'file_path' in document.metadata.keys():
                    del document.metadata['file_path']

                if not document.text:
                    logging.error(f"Found empty document text in [{title}] with [{file_path}]")
                    continue
                document.text.replace('', '')

                document.metadata.update({
                    'document_type': DOCUMENT_TYPES.ARTICLE.value,
                    'title': title,
                    'authors': "",
                    'pdf_link': "",
                    'release_date': "",
                })

            logging.warning(f"Failed to find metadata for (no row match) [{file_path}], adding only title")
        # logging.info(f"Loaded metadata for [{title}] with [{file_path}]")
        save_successful_load_to_csv(documents[0], csv_filename='all_discourse_articles.csv', fieldnames=['title', 'authors', 'pdf_link', 'release_date'])
        return documents
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path], articles_aggregates_path: Union[str, Path] = f"{root_dir}/datasets/evaluation_data/merged_articles.csv", overwrite=False, num_files: int = None):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []

    latest_df = pd.read_csv(articles_aggregates_path)
    current_df = pd.read_csv(f"{root_dir}/pipeline_storage/all_discourse_articles.csv")
    paper_details_df = compute_new_entries(latest_df=latest_df, current_df=current_df, left_key='Link', overwrite=overwrite)

    partial_load_single_pdf = partial(load_single_pdf, paper_details_df=paper_details_df, current_df=current_df if not overwrite else pd.DataFrame(columns=['title', 'authors', 'pdf_link', 'release_date']))

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
                if not documents:
                    continue
                all_documents.extend(documents)
                pdf_loaded_count += 1
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}: [{e}]")
                os.remove(pdf_file)

    logging.info(f"Successfully loaded [{pdf_loaded_count}] documents from [{DOCUMENT_TYPES.ARTICLE.value}] Discourse files.")
    return all_documents


if __name__ == '__main__':
    # Example usage
    pdf_directory = f"{root_dir}/datasets/evaluation_data/articles_2023-12-05"
    articles_aggregates_path = f"{root_dir}/datasets/evaluation_data/merged_articles.csv"  # Updated path
    load_pdfs(pdf_directory, articles_aggregates_path)
