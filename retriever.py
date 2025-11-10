import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PIL import Image
import pytesseract
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def load_files(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        text = ""
        if fname.lower().endswith(".pdf"):
            reader = PdfReader(path)
            text = "".join([p.extract_text() or "" for p in reader.pages])
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(path)
            text = pytesseract.image_to_string(image)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def chunk_files(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def hybrid_retriever(chunks, persist_dir="./chroma_db", bm25_weight=0.5, semantic_weight=0.5, k=3):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    semantic_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight]
    )

    return ensemble_retriever
