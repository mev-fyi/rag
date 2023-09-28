from rag.Llama_index_sandbox import root_directory

from dotenv import load_dotenv
load_dotenv()

# Append the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

root_dir = root_directory()
mev_fyi_dir = f"{root_dir}/../mev.fyi/"
research_papers_dir = f"{mev_fyi_dir}/data/paper_details.csv"
pdfs_dir = f"{mev_fyi_dir}/data/papers_pdf_downloads/"
