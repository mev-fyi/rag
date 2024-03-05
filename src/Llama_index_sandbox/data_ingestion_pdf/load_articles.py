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

        paper_row = paper_details_df[paper_details_df['title'] == title]  # if there is a match it means that we need to add that title for which we have metadata

        if not paper_row.empty and not is_in_current_df:
            assert paper_row.iloc[0]['title'] != np.nan, f"Title is NaN for {paper_row.iloc[0]['article']}"
            assert paper_row.iloc[0]['authors'] != np.nan, f"authors is NaN for {paper_row.iloc[0]['article']}"
            assert paper_row.iloc[0]['article'] != np.nan, f"pdf_link is NaN for {paper_row.iloc[0]['article']}"
            assert paper_row.iloc[0]['release_date'] != np.nan, f"release_date is NaN for {paper_row.iloc[0]['article']}"

            # Update metadata
            for document in documents:
                if 'file_path' in document.metadata.keys():
                    del document.metadata['file_path']

                if '<|endoftext|>' in document.text:
                    logging.error(f"Found <|endoftext|> in {title} with {file_path}")
                document.text.replace('<|endoftext|>', '')

                document.metadata.update({
                    'document_type': DOCUMENT_TYPES.ARTICLE.value,
                    'title': title,
                    'authors': str(paper_row.iloc[0]['authors']),
                    'pdf_link': str(paper_row.iloc[0]['article']),
                    'release_date': str(paper_row.iloc[0]['release_date']),
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
            # NOTE 2024-03-05: now that we only do incremental updates, a lack of match no longer means that we could not find the metadata,
            #   but rather means that we processed the file
            logging.warning(f"Failed to find metadata for [{file_path}], adding only title")
        save_successful_load_to_csv(documents[0], csv_filename='articles.csv', fieldnames=['title', 'authors', 'pdf_link', 'release_date'])
        return documents
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        # Find the corresponding row in the DataFrame
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path], overwrite=False, num_files: int = None):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []

    latest_df = pd.read_csv(f"{root_dir}/datasets/evaluation_data/articles_updated.csv")
    current_df = pd.read_csv(f"{root_dir}/pipeline_storage/articles.csv")
    paper_details_df = compute_new_entries(latest_df=latest_df, current_df=current_df, left_key='article', overwrite=overwrite)

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
                # os.remove(pdf_file)

    logging.info(f"Successfully loaded [{pdf_loaded_count}] documents from [{DOCUMENT_TYPES.ARTICLE.value}] files.")
    return all_documents


if __name__ == '__main__':
    # Example usage
    pdf_directory = f"{root_dir}/datasets/evaluation_data/articles_2023-12-05"
    load_pdfs(pdf_directory)
