import os
import sys

from rag.Llama_index_sandbox.utils import root_directory
# Append the parent directory to sys.path

from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
load_dotenv()

root_dir = root_directory()
mev_fyi_dir = f"{root_dir}/../mev.fyi/"
RESEARCH_PAPER_CSV = f"{mev_fyi_dir}/data/paper_details.csv"
PDF_DIRECTORY = f"{root_dir}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-10-05/"
YOUTUBE_VIDEO_DIRECTORY = f"{root_dir}/datasets/evaluation_data/diarized_youtube_content_2023-10-06/"
index_dir = f"{root_dir}/.storage/research_pdf/"
