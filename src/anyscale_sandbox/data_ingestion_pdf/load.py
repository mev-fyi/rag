import concurrent.futures
import logging
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox.constants import DOCUMENT_TYPES
from src.Llama_index_sandbox.utils.utils import timeit
from src.anyscale_sandbox import root_dir, mev_fyi_dir, research_papers_dir


@timeit
def fetch_pdf_list(num_papers=None):

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(research_papers_dir)

    # Append '.pdf' to the links that contain 'arxiv' and subselect all the ones which contain '.pdf'
    df['pdf_link'] = df['pdf_link'].apply(lambda link: link + '.pdf' if 'arxiv' in link else link)
    pdf_links = df.loc[df['pdf_link'].str.contains('.pdf'), 'pdf_link'].tolist()

    # If num_papers is specified, subset the list of pdf_links
    if num_papers is not None:
        pdf_links = pdf_links[:num_papers]

    # Directory to save the PDF files
    save_dir = f"{root_dir}/data/downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)
    return pdf_links, save_dir, df


@timeit
def download_pdfs(pdf_links, save_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(partial(download_pdf, save_dir=save_dir), pdf_links)


def download_pdf(link, save_dir):
    # Extract the file name from the link
    file_name = link.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)

    # Check if the file has already been downloaded locally
    if os.path.exists(file_path):
        # logging.info(f"{file_name} already exists locally. Skipping download.")
        return

    # If not, download the file with retries
    retries = 3
    for _ in range(retries):
        try:
            logging.info(f"requesting pdf {link}")
            time.sleep(random.uniform(1, 5))
            # Send an HTTP request to the server and save the PDF file
            response = requests.get(link)
            response.raise_for_status()

            # Check if the content type is PDF
            if response.headers['Content-Type'] == 'application/pdf':
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Successfully downloaded {file_name}")
            else:
                logging.info(f"Failed to download a valid PDF file from {link}")

            break

        except requests.exceptions.RequestException as e:
            logging.info(f"Failed to download {link}, retrying...: {e}")
            time.sleep(0.25)  # Sleep before retrying
    else:
        logging.info(f"Failed to download {file_name} after {retries} retries.")


def load_single_pdf(paper_details_df, file_path, loader=PyMuPDFReader()):
    try:
        documents = loader.load(file_path=file_path)

        # Update 'file_path' metadata and add additional metadata
        for document in documents:
            if 'file_path' in document.metadata:
                del document.metadata['file_path']

            # Find the corresponding row in the DataFrame
            title = os.path.basename(file_path).replace('.pdf', '')
            paper_row = paper_details_df[paper_details_df['title'] == title]

            if not paper_row.empty:
                # Update metadata
                document.metadata.update({
                    'title': paper_row.iloc[0]['title'],
                    'authors': paper_row.iloc[0]['authors'],
                    'pdf_link': paper_row.iloc[0]['pdf_link'],
                    'release_date': paper_row.iloc[0]['release_date']
                })
            # TODO 2023-09-27: add relevance score as metadata. The score will be highest for research papers, ethresear.ch posts.
            #   It will be high (highest too? TBD.) for talks and conferences in YouTube video format
            #   It will be relatively lower for podcasts, tweets, and less formal content.
        return documents
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path]):
    # Convert directory_path to a Path object if it is not already
    # logging.info("Loading PDFs")
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    research_papers_path = f"{mev_fyi_dir}/data/paper_details.csv"

    paper_details_df = pd.read_csv(research_papers_path)
    partial_load_single_pdf = partial(load_single_pdf, paper_details_df=paper_details_df)
    pdf_loaded_count = 0

    # Using ThreadPoolExecutor to load PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map over all PDF files in the directory
        futures = {executor.submit(partial_load_single_pdf, file_path=pdf_file): pdf_file for pdf_file in directory_path.glob("*.pdf")}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                pdf_loaded_count += 1
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}: {e}")
    logging.info(f"Successfully loaded {pdf_loaded_count} [{DOCUMENT_TYPES.RESEARCH_PAPER.value}] documents.")
    return all_documents
