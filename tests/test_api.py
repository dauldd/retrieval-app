import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import api
import app as app_module

client = TestClient(api)

TEST_CONTENT = """
There were 17 people on the ship yesterday.
The captain's name was Jack.
The ship carried 150 livestock units.
"""

@pytest.fixture(autouse=True)
def reset_app_state():
    """reset the app state before and after a test"""
    data_dir = Path("data")
    if data_dir.exists():
        for file in data_dir.glob("*.txt"):
            file.unlink()

    app_module.docs = []
    app_module.chunks = []
    app_module.retriever = None
    app_module.qa_chain = None
    yield
    if data_dir.exists():
        for file in data_dir.glob("*.txt"):
            file.unlink()

def test_query_without_documents():
    """test that querying without uploading docs returns an error"""
    response = client.post("/api/query", json={"query": "how many humans were there on the ship?"})
    assert response.status_code == 500
    assert "error" in response.json()

def test_upload_file():
    """test uploading a txt file """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_CONTENT)
        temp_path = f.name
    
    try:
        with open(temp_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": ("ship_info.txt", f, "text/plain")}
            )
        if response.status_code != 200:
            print(f"upload failed: {response.json()}")
        assert response.status_code == 200
        assert "uploaded and indexed" in response.json()["message"]

        saved_file = Path("data/ship_info.txt")
        assert saved_file.exists()
    
    finally:
        Path(temp_path).unlink()


def test_query_after_upload():
    """test querying after uploading txt"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_CONTENT)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            upload_response = client.post(
                "/api/upload",
                files={"file": ("ship_info.txt", f, "text/plain")}
            )

        assert upload_response.status_code == 200

        query_response = client.post(
            "/api/query",
            json={"query": "How many livestock units were there on the ship yesterday?"}
        )


        assert query_response.status_code == 200
        data = query_response.json()
        print("answer:", data["answer"])
        print("sources:", data["sources"])

        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

    finally:
        Path(temp_path).unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
