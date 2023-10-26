from llama_index.text_splitter import SentenceSplitter

from src.Llama_index_sandbox.utils import timeit


def get_chunk_overlap(TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE):
    return int(TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE/100 * TEXT_SPLITTER_CHUNK_SIZE)


@timeit
def chunk_documents(documents, TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE, splitter_fn=None, chunk_overlap=None):
    # NOTE: we expect semantical splitter methods to perform best (e.g. sentence splitter)
    if splitter_fn is None:
        splitter_fn = SentenceSplitter  # TODO 2023-09-25: The chosen text splitter should be a hyperparameter we can tune.
    if chunk_overlap is None:
        chunk_overlap = get_chunk_overlap(TEXT_SPLITTER_CHUNK_SIZE=TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE=TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE)
    text_chunks, doc_idxs = chunk_single_document(documents, TEXT_SPLITTER_CHUNK_SIZE, splitter_fn=splitter_fn, chunk_overlap=chunk_overlap, separator="\n")
    return text_chunks, doc_idxs


@timeit
def chunk_single_document(documents, TEXT_SPLITTER_CHUNK_SIZE, splitter_fn, chunk_overlap, separator="\n"):
    # TODO 2023-09-25: Chunk overlap is one hyperparameter we can tune
    text_splitter = splitter_fn(chunk_size=TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=chunk_overlap)

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    return text_chunks, doc_idxs
