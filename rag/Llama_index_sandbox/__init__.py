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
research_papers_dir = f"{mev_fyi_dir}/data/paper_details.csv"
pdfs_dir = f"{root_dir}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-10-05/"
video_transcripts_dir = f"{root_dir}/datasets/evaluation_data/diarized_youtube_content_2023-10-04/"
index_dir = f"{root_dir}/.storage/research_pdf/"
