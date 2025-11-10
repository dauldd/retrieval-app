from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import app

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api.mount("/static", StaticFiles(directory="static"), name="static")

upload_dir = Path("data")
upload_dir.mkdir(parents=True, exist_ok=True)

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