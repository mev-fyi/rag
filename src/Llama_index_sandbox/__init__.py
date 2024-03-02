import os
import sys

from src.Llama_index_sandbox.evaluation.config import Config

# Append the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.Llama_index_sandbox.utils.utils import root_directory

from dotenv import load_dotenv
load_dotenv()

root_dir = root_directory()
mev_fyi_dir = f"{root_dir}/../mev.fyi/"
RESEARCH_PAPER_CSV = f"{mev_fyi_dir}/data/paper_details.csv"
# PDF_DIRECTORY = f"{root_dir}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-10-05/"
PDF_DIRECTORY = f"{root_dir}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-11-21/"
ARTICLES_DIRECTORY = f"{root_dir}/datasets/evaluation_data/articles_2023-12-05"
DISCOURSE_ARTICLES_DIRECTORY = f"{root_dir}/datasets/evaluation_data/articles_discourse_2024_03_01"
FLASHBOTS_DOCS_DIRECTORY = f"{root_dir}/datasets/evaluation_data/flashbots_docs_2024_01_07"
ETHEREUM_ORG_DOCS_DIRECTORY = f"{root_dir}/datasets/evaluation_data/ethereum_org_content_docs_2024_01_07"
YOUTUBE_VIDEO_DIRECTORY = f"{root_dir}/datasets/evaluation_data/diarized_youtube_content_2023-10-06/"
config_instance = Config()
output_dir = config_instance.get_index_output_dir()
