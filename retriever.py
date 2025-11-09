import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_files(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            reader = PdfReader(path)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def chunk_files(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def chunks_to_embeddings(chunks, persist_dir="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    return vectordb.as_retriever(search_kwargs={"k": 3})

