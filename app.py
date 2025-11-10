import os
from retriever import load_files, chunk_files, hybrid_retriever
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

docs = []
chunks = []
retriever = None
qa_chain = None

def create_chain():
    global qa_chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def initialize():
    global docs, chunks, retriever
    docs = load_files("data")
    chunks = chunk_files(docs)
    retriever = hybrid_retriever(chunks)
    create_chain()

def ingest_file(file_path):
    global docs, chunks, retriever

    new_docs = load_files(os.path.dirname(file_path))
    new_doc = [d for d in new_docs if d.metadata["source"] == os.path.basename(file_path)]
    if not new_doc:
        raise ValueError(f"no text extracted from {file_path}")
    new_chunks = chunk_files(new_doc)
    chunks.extend(new_chunks)

    retriever = hybrid_retriever(chunks)
    create_chain()
    print(f"indexed {len(new_chunks)} new chunks from {file_path}")

def ask(query: str):
    if not qa_chain:
        raise ValueError("no documents indexed yet")
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]
    return answer, sources


# while True:
#     query = input("ask a question:")
#     if query.lower() == "exit":
#         break
#     result = qa_chain.invoke({"query": query})
#     print("answer:", result["result"])
#     print("sources:")
#     for i, doc in enumerate(result["source_documents"], 1):
#         print(f" - Chunk {i}: {doc.metadata.get('source', 'Unknown')}")