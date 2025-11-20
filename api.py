from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
import shutil
import app


@asynccontextmanager
async def lifespan(_: FastAPI):
    chroma_db_path = Path("./chroma_db")
    data_path = Path("data")

    has_chroma_data = chroma_db_path.exists() and any(chroma_db_path.iterdir())
    has_data_files = data_path.exists() and any(
        f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.png', '.jpg', '.jpeg']
        for f in data_path.iterdir()
    )

    if has_chroma_data and has_data_files:
        try:
            print("found existing data and initializing retriever...")
            app.initialize()
            print(f"successfully initialized with {len(app.docs)} documents and {len(app.chunks)} chunks")
        except Exception as e:
            print(f"couldn't initialize from existing data: {e}")
    yield

api = FastAPI(lifespan=lifespan)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api.mount("/static", StaticFiles(directory="static"), name="static")

@api.get("/")
async def root():
    return FileResponse("static/index.html")

upload_dir = Path("data")
upload_dir.mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@api.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    file_path = upload_dir / Path(file.filename).name
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()

    try:
        app.ingest_file(file_path)
        return {"message": f"{file.filename} uploaded and indexed."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@api.post("/api/query")
async def query_docs(request: QueryRequest):
    try:
        answer, sources = app.ask(request.query)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
