
"""
TODO we need to clean the generated transcripts. we can either do it at transcript generation time or afterwards.
# TODO 2023-10-29: upgrading transcripts to identify who are the speakers would have tremendous value. Likely that can be done by passing each transcript to an LLM

items to clean:
"L Two": "L2",
"L Two s": "L2s",
"L Three": "L3s",
"Arbitram": "Arbitrum",
"web three": "web3",
"SWAV": "SUAVE",
"One Inc": "1inch",
"Pepsi": "PEPC",
"""

import os

from src.Llama_index_sandbox import root_directory


def correct_typos_in_files(log=True):

    """
    Correct specific typos in .txt files under a given directory.

    Args:
    - root_dir (str): Path to the root directory where search begins.
    """
    root_dir = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"

    # Dictionary of typos and their corrections
    typo_dict = {
        "L Two": "L2",
        "L Two s": "L2s",
        "L Three": "L3s",
        "Arbitram": "Arbitrum",
        "web three": "web3",
        "SWAV": "SUAVE",
        "One Inc": "1inch",
        "Pepsi": "PEPC",
        "MVV": "MEV",
    }

    # Walk through root_dir
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith("_diarized_content_processed_diarized.txt"):
                file_path = os.path.join(dirpath, fname)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    for typo, correction in typo_dict.items():
                        content = content.replace(typo, correction)
                        if log:
                            print("Corrected typo: ", typo, " -> ", correction, " in ", file_path, "\n")

                # Write corrected content back to file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)


if __name__ == "__main__":
    correct_typos_in_files()

