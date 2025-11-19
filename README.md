# Document Query Application

Search application that lets a user upload files and ask questions about their contents. The system uses hybrid retrieval combining BM25 keyword matching with semantic embeddings with LLM-based answer generation.

The purpose of this project was to analyze the performance and interaction of different retrieval methods in document search systems, to explore how hybrid retrieval combining keyword-based and semantic approaches affects result quality, and to allow real-time document queries via FastAPI backend interface.

## Architecture

The application is organized into three components. The [retriever.py](retriever.py) module handles document loading and processing. It reads PDF files (PyPDF2), plain text, and images (pytesseract OCR - Tesseract required) through OCR, then splits the content into 800-character chunks with 100-character overlap. The `HybridRetrieverManager` class enables incremental document ingestion - new chunks are added directly to the existing ChromaDB vector database via `add_documents()` without re-embedding previous documents, significantly improving upload performance. The [app.py](app.py) module manages the application state, tracking uploaded documents and their chunks while coordinating the retriever updates and query execution. Then [api.py](api.py) provides the FastAPI web server with upload and query endpoints with frontend in static/.

```
├── app.py               # core logic (state management, QA chain)
├── retriever.py         # hybrid retriever (BM25 + embeddings)
├── api.py               # FastAPI backend
├── requirements.txt     # dependencies
├── README.md            # documentation
├── .gitignore
├── data/                # data for indexing
├── static/              # frontend
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── tests/
    ├── test_api.py      # API tests with randomized queries
    └── compare_scores.py # score comparison utility
```

The system uses a hybrid approach that combines two retrieval methods. BM25 handles traditional keyword matching, while a semantic retriever uses all-MiniLM-L6-v2 embeddings to find semantically similar content. Each method retrieves three results, weighted equally at 50% each, then merged through an ensemble retriever. Vector embeddings are stored in a local ChromaDB database at `./chroma_db/`.

When a new file is uploaded, only the new chunks are embedded and added to ChromaDB via `HybridRetrieverManager.add_documents()`. The BM25 index is rebuilt in-memory with all accumulated chunks, and the ensemble retriever is updated. This approach avoids re-processing existing documents, making subsequent uploads significantly faster.

## API Endpoints

### POST /api/upload

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@document.txt"
```

Uploads file to data/ directory, extracts text by type, chunks content, adds to collection, and rebuilds retriever with all chunks.

### POST /api/query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

Returns answer and source documents. Hybrid retriever processes query and passes relevant chunks to LLM (gemini-2.5-flash) for answer generation.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_api_key
uvicorn api:api --reload
```

The application will be available at http://localhost:8000/

## Testing

```bash
python tests/test_api.py           # error handling, upload, query with randomized selection and cosine similarity
python tests/compare_scores.py     # interactive BM25 vs semantic comparison
```
