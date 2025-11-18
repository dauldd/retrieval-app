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
import os
import shutil

class HybridRetrieverManager:
    
    def __init__(self, persist_dir="./chroma_db", k=3, bm25_weight=0.5, semantic_weight=0.5):
        self.persist_dir = persist_dir
        self.k = k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self.vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self.vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        
        self.all_chunks = []
        self.bm25_retriever = None
        self.ensemble_retriever = None
    
    def add_documents(self, new_chunks):
        if not new_chunks:
            return
        
        self.vectordb.add_documents(new_chunks)

        self.all_chunks.extend(new_chunks)
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        self.bm25_retriever.k = self.k

        self._rebuild_ensemble()
    
    def _rebuild_ensemble(self):
        semantic_retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, semantic_retriever],
            weights=[self.bm25_weight, self.semantic_weight]
        )

    def get_retriever(self):
        return self.ensemble_retriever

    def get_chunk_count(self):
        return len(self.all_chunks)
    
    def clear(self):
        self.all_chunks = []
        self.bm25_retriever = None
        self.ensemble_retriever = None
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

def load_files(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        text = ""
        if fname.lower().endswith(".pdf"):
            reader = PdfReader(path)
            text = "".join([p.extract_text() or "" for p in reader.pages])
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(path)
            text = pytesseract.image_to_string(image)
        elif fname.lower().endswith('.txt'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
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
