import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Configuration
DB_DIR = "chroma_db"

FETCH_K = 30

TOP_K = 5


# Embedding Model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Database

vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Tokenizer

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )
# Dense Retrieval

def dense_search(query):

    return vectordb.max_marginal_relevance_search(
        query=query,
        k=FETCH_K,
        fetch_k=50
    )

# BM25 Reranking

def bm25_rerank(query, documents):

    corpus = [
        doc.page_content
        for doc in documents
    ]

    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for _, doc in ranked
    ]

# CrossEncoder Reranking

def crossencoder_rerank(query, documents):

    pairs = [
        (
            query,
            doc.page_content
        )
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for _, doc in ranked[:TOP_K]
    ]

# Build Context

def build_context(chunks):

    context = ""

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        context += (
            f"Source : {source}\n"
            f"Page : {page}\n\n"
            f"{chunk.page_content}\n"
            "\n---------------------------------------------\n\n"
        )

    return context

# Main Retrieval Function

def retrieve(query):

    dense_results = dense_search(query)

    bm25_results = bm25_rerank(
        query,
        dense_results
    )

    final_chunks = crossencoder_rerank(
        query,
        bm25_results
    )

    return build_context(
        final_chunks
    )