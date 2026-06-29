import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_FOLDER = "data"
DB_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)


vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

existing_sources = set()

try:

    data = vectordb.get()

    if data["metadatas"]:

        for metadata in data["metadatas"]:

            existing_sources.add(
                metadata["source"]
            )

except:
    pass
# Process PDFs

new_chunks = []

print("\nSearching for PDFs...\n")

for file in os.listdir(DATA_FOLDER):

    if not file.lower().endswith(".pdf"):
        continue

    if file in existing_sources:

        print(f"Skipping {file} (Already Ingested)")

        continue

    pdf_path = os.path.join(DATA_FOLDER, file)

    print(f"\nLoading {file}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} semantic chunks")

    # Add metadata

    for i, chunk in enumerate(chunks):

        chunk.metadata["source"] = file

        chunk.metadata["page"] = chunk.metadata.get(
            "page",
            -1
        )

        chunk.metadata["chunk_id"] = hashlib.md5(
            (
                file + str(i)
            ).encode()
        ).hexdigest()

    new_chunks.extend(chunks)

# Store Chunks

if len(new_chunks) > 0:

    print("\nAdding new chunks to ChromaDB...")

    vectordb.add_documents(
        new_chunks
    )

    print(
        f"Added {len(new_chunks)} chunks."
    )

else:

    print(
        "\nNo new PDFs found."
    )

print("\nIngestion Complete!")