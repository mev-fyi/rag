import concurrent.futures
import logging
import os
import random
import time
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd
import pikepdf
import requests
from fitz import fitz
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pypdf import PdfReader

from rag.Llama_index_sandbox import root_dir, RESEARCH_PAPER_CSV
from rag.Llama_index_sandbox.constants import *
from rag.Llama_index_sandbox.utils import timeit


def get_pdf_details(response: requests.Response) -> dict:
    try:
        response.raise_for_status()

        # Step 5: Using PyPDF2 to get the PDF details
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            info = reader.metadata

            paper_title = info.title
            if info.title:  # Try getting details with PyPDF2
                try:
                    paper_title = info.title
                except Exception as e:
                    logging.info(f"Could not retrieve details with [PyPDF] from: {e}")
            if not paper_title:  # If details retrieval still fails, try with PyMuPDF
                try:
                    f.seek(0)
                    doc = fitz.open(f)
                    info = doc.metadata
                    paper_title = info['title']
                except Exception as e:
                    logging.info(f"Could not retrieve details with [PyMuPDF] from: {e}")

            if not paper_title:  # If details retrieval still fails, try with pdfminer
                try:
                    f.seek(0)
                    parser = PDFParser(f)
                    doc = PDFDocument(parser)
                    info = doc.info[0]
                    paper_title = info['Title'].decode('utf-8', 'ignore') if 'Title' in info else ''
                except Exception as e:
                    logging.info(f"Could not retrieve details with [pdfminer] from: {e}")

            if not paper_title:  # If details retrieval still fails, try with pikepdf
                try:
                    f.seek(0)
                    doc = pikepdf.open(f)
                    info = doc.open_metadata()
                    paper_title = info['Title'] if 'Title' in info else ''
                except Exception as e:
                    logging.info(f"Could not retrieve details with [pikepdf] from: {e}")

            # Extract creation date and format it to "yyyy-mm-dd"
            logging.info('\n', info)

            # Creating and returning the details dictionary
            details = {
                "title": paper_title,
            }
            logging.info(f"Retrieved: {paper_title}\n\n")
    except Exception as e:
        # Creating and returning the details dictionary
        details = {
            "title": None,
        }
        logging.info(f"Could not retrieve details from: {e}")
    return details

@timeit
def fetch_pdf_list(num_papers=None):

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(RESEARCH_PAPER_CSV)

    # Append '.pdf' to the links that contain 'arxiv' and subselect all the ones which contain '.pdf'
    df['pdf_link'] = df['pdf_link'].apply(lambda link: link + '.pdf' if 'arxiv' in link else link)
    pdf_links = df.loc[df['pdf_link'].str.contains('.pdf'), 'pdf_link'].tolist()

    # If num_papers is specified, subset the list of pdf_links
    if num_papers is not None:
        pdf_links = pdf_links[:num_papers]

    # Directory to save the PDF files
    save_dir = f"{root_dir}/data/downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)
    return pdf_links, save_dir


@timeit
def download_pdfs(pdf_links, save_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(partial(download_pdf, save_dir=save_dir), pdf_links)


def download_pdf(link, save_dir, optional_file_name=None):
    # Extract the file name from the link
    if not optional_file_name:
        file_name = link.split("/")[-1]
    else:
        file_name = optional_file_name
    file_path = os.path.join(save_dir, file_name)

    # Check if the file has already been downloaded locally
    if os.path.exists(file_path):
        return file_path

    # If not, download the file with retries
    retries = 3
    for _ in range(retries):
        try:
            logging.info(f"Requesting PDF {link}")
            time.sleep(random.uniform(1, 5))
            # Send an HTTP request to the server and save the PDF file
            response = requests.get(link)
            response.raise_for_status()

            if not optional_file_name:
                file_name = get_pdf_details(response)['title'] + '.pdf'

            # Check if the content type is PDF
            if response.headers['Content-Type'] == 'application/pdf':
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Successfully downloaded {file_name}")
                return file_path
            else:
                logging.info(f"Failed to download a valid PDF file from {link}")

        except requests.exceptions.RequestException as e:
            logging.info(f"Failed to download {link}, retrying...: {e}")
            time.sleep(0.25)  # Sleep before retrying
    else:
        logging.info(f"Failed to download {file_name} after {retries} retries.")
        return None


def load_single_pdf(paper_details_df, file_path, loader=PyMuPDFReader()):
    try:
        documents = loader.load(file_path=file_path)
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        # Find the corresponding row in the DataFrame
        title = os.path.basename(file_path).replace('.pdf', '')
        paper_row = paper_details_df[paper_details_df['title'] == title]

        if not paper_row.empty:
            pdf_link = paper_row.iloc[0]['pdf_link']
            pdf_link += '.pdf' if not 'pdf' in pdf_link else ''
            save_dir = Path(file_path).parent
            new_file_path = download_pdf(pdf_link, save_dir, title + '.pdf')
            if new_file_path:
                documents = loader.load(file_path=new_file_path)
            else:
                return []

    # Update 'file_path' metadata and add additional metadata
    for document in documents:
        if 'file_path' in document.metadata.keys():
            del document.metadata['file_path']

        # Find the corresponding row in the DataFrame
        title = os.path.basename(file_path).replace('.pdf', '')
        paper_row = paper_details_df[paper_details_df['title'] == title]

        if not paper_row.empty:
            # Update metadata
            document.metadata.update({
                'document_type': DOCUMENT_TYPES.RESEARCH_PAPER.value,
                'title': paper_row.iloc[0]['title'],
                'authors': paper_row.iloc[0]['authors'],
                'pdf_link': paper_row.iloc[0]['pdf_link'],
                # TODO 2023-10-08: we might want to limit date to yyyy-mm only  https://docs.pinecone.io/docs/metadata-filtering
                'release_date': paper_row.iloc[0]['release_date']
            })
            # TODO 2023-09-27: add relevance score as metadata. The score will be highest for research papers, ethresear.ch posts.
            #   It will be high (highest too? TBD.) for talks and conferences in YouTube video format
            #   It will be relatively lower for podcasts, tweets, and less formal content.

    return documents


@timeit
def load_pdfs(directory_path: Union[str, Path]):
    # Convert directory_path to a Path object if it is not already
    # logging.info("Loading PDFs")
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    research_papers_path = f"{root_dir}/datasets/evaluation_data/paper_details.csv"

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
                logging.info(f"Failed to process {pdf_file}, removing file: {e}")
                os.remove(pdf_file)
    logging.info(f"Successfully loaded {pdf_loaded_count} documents from PDF files.")
    return all_documents
