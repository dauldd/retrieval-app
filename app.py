import os
from retriever import load_files, chunk_files, chunks_to_embeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

docs = load_files("data")
chunks = chunk_files(docs)
retriever = chunks_to_embeddings(chunks)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    query = input("ask a question:")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke({"query": query})
    print("answer:", result["result"])
    print("sources:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f" - Chunk {i}: {doc.metadata.get('source', 'Unknown')}")