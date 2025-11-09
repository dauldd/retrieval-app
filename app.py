import os
from retriever import load_files, chunk_files, chunks_to_embeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

docs = load_files("data")
chunks = chunk_files(docs)
retriever = chunks_to_embeddings(chunks)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    query = input("ask a question:")
    result = qa_chain.invoke({"query": query})
    print("answer:", result["result"])