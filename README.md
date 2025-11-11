# Document Query Application

Search application that lets a user upload files and ask questions about their contents. The system uses hybrid retrieval combining BM25 keyword matching with semantic embeddings with LLM-based answer generation.

The purpose of this project was to analyze the performance and interaction of different retrieval methods in document search systems, to explore how hybrid retrieval combining keyword-based and semantic approaches affects result quality, and to allow real-time document queries via FastAPI backend interface.

## Architecture

The application is organized into three components. The [retriever.py](retriever.py) module handles document loading and processing. It reads PDF files, plain text, and images through OCR, then splits the content into 800-character chunks and builds the hybrid retriever. The [app.py](app.py) module manages the application state, tracking uploaded documents and their chunks while coordinating the retriever updates and query execution. Then [api.py](api.py) provides the FastAPI web server with upload and query endpoints with frontend in static/.

```
├── app.py               # core logic (state management, QA chain)
├── retriever.py         # hybrid retriever (BM25 + embeddings)
├── api.py               # FastAPI backend
├── requirements.txt     # dependencies
├── README.md            # documentation
├── .gitignore
├── static/              # frontend
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── tests/
    ├── test_api.py      # API tests with randomized queries
    └── compare_scores.py # score comparison utility
```

The system uses a hybrid approach that combines two retrieval methods. BM25 handles traditional keyword matching, while a semantic retriever uses MiniLM embeddings to find semantically similar content. Each method retrieves three results, weighted equally at 50% each, then merged through an ensemble retriever. Vector embeddings are stored in a local ChromaDB database at `./chroma_db/`.

When the new file is uploaded, the system extracts the text, chunks it, adds the chunks to the existing collection, then recreates the retriever from scratch using accumulated chunks. This triggers recreation of the QA chain with the updated retriever.

## API Endpoints

### POST /api/upload

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@document.txt"
```

Returns a confirmation message upon successful upload and indexing. The endpoint saves the file to the data/ directory, extracts text according to the file type (PDF, plain text or image), chunks the content, and adds these chunks to the existing collection. The retriever is then rebuilt from scratch using all available chunks, and a new QA chain is created with the updated retriever.

### POST /api/query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

Returns an answer and a list of source documents. The query is processed by the hybrid retriever, which returns relevant chunks from both bm25 and semantic searches. These chunks are passed to the LLM (gemini-2.5-flash for this project) as context, which generates a natural language answer. The response includes both the answer text and the filenames of the source documents being used.

## Setup

To complete the setup, install the required dependencies, configure the API key and start the development server.

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_api_key
uvicorn api:api --reload
```

The application will be available at http://localhost:8000/

## Configuration

The system processes PDF files using PyPDF2 for text extraction, plain text files through direct reading, and images (png, jpg, jpeg) via pytesseract OCR (Tesseract OCR should be installed in the system for optical character recognition from images).

The retriever is configured in the `hybrid_retriever()` function with a chunk size of 800 characters and 100-character overlap. Both bm25 and semantic retrievers are weighted equally at 0.5 and return three results each. The semantic component uses the all-MiniLM-L6-v2 embedding model from HuggingFace.

## Testing

The first test (1) checks error handling, file upload functionality and query functionality after upload with randomized query selection.

```bash
python tests/test_api.py
```

The second test (2) starts an interactive comparison of two retrieval methods - BM25 and semantic embeddings from all-MiniLM-L6-v2

```bash
python tests/compare_scores.py
```
