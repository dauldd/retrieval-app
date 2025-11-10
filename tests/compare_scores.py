import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from retriever import load_files, chunk_files
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
docs = load_files(data_dir)

chunks = chunk_files(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5

print("score comparison")

while True:
    query = input("\nquery: ")
    bm25_docs = bm25.invoke(query)
    sem_results = vectordb.similarity_search_with_score(query, k=5)

    print(f"\n{'bm25':<150}")

    for i, doc in enumerate(bm25_docs, 1):
        preview = doc.page_content[:250].replace('\n', ' ')
        print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
        print(f"   {preview}...")

    print(f"\n{'semantic':<150}")
    for i, (doc, score) in enumerate(sem_results, 1):
        preview = doc.page_content[:250].replace('\n', ' ')
        print(f"{i}. {doc.metadata.get('source', 'Unknown')} (score: {score:.3f})")
        print(f"   {preview}...")