from llama_index.text_splitter import SentenceSplitter

from rag.Llama_index_sandbox.utils import timeit


def get_chunk_overlap(chunk_size):
    return int(0.15 * chunk_size)  # TODO 2023-09-26: tune the chunk_size


@timeit
def chunk_documents(documents, chunk_size, splitter_fn=None, chunk_overlap=None):
    # NOTE: we expect semantical splitter methods to perform best (e.g. sentence splitter)
    if splitter_fn is None:
        splitter_fn = SentenceSplitter  # TODO 2023-09-25: The chosen text splitter should be a hyperparameter we can tune.
    if chunk_overlap is None:
        chunk_overlap = get_chunk_overlap(chunk_size)
    text_chunks, doc_idxs = chunk_single_document(documents, chunk_size, splitter_fn=splitter_fn, chunk_overlap=chunk_overlap, separator="\n")
    return text_chunks, doc_idxs


@timeit
def chunk_single_document(documents, chunk_size, splitter_fn, chunk_overlap, separator="\n"):
    # TODO 2023-09-25: Chunk overlap is one hyperparameter we can tune
    text_splitter = splitter_fn(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    return text_chunks, doc_idxs
