import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import api
import app as app_module

client = TestClient(api)

DOC_A ="""
There were 17 people on the ship yesterday.
The captain's name was Jack.
The ship carried 150 livestock units.
Out of 150 livestock units - 67 cows, 22 chicken, 34 sheeps, 27 goats.
"""

DOC_B = """
The warehouse contains 89 employees.
The warehouse manager's name is Sarah.
The warehouse stores 3400 boxes total.
Out of 3400 boxes, 1200 contain electronics, 800 contain clothing, 900 contain food, 500 contain books.
"""

DOC_C = """
The factory has 52 machines in operation.
The factory supervisor's name is Chen.
The factory produces 12000 units per day.
Out of 12000 units, 4500 are red widgets, 3200 are blue widgets, 2800 are green widgets, 1500 are yellow widgets.
"""

@pytest.fixture(autouse=True)
def reset_app_state():
    """resetting the app state before and after"""
    data_dir = Path("data")
    if data_dir.exists():
        for file in data_dir.glob("*.txt"):
            file.unlink()

    app_module.docs = []
    app_module.chunks = []
    app_module.retriever_manager = app_module.HybridRetrieverManager()
    app_module.qa_chain = None
    yield
    if data_dir.exists():
        for file in data_dir.glob("*.txt"):
            file.unlink()

def upload_document(content: str, filename: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        start_time = time.time()
        with open(temp_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": (filename, f, "text/plain")}
            )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert "uploaded and indexed" in response.json()["message"]

        return elapsed_time
    finally:
        Path(temp_path).unlink()

def test_incremental_indexing_performance():
    """testing that subsequent uploads don't reprocess previous documents"""

    time_a = upload_document(DOC_A, "doc_a.txt")
    print(f"DOC A uplod time: {time_a: 3f}s")

    time_b = upload_document(DOC_B, "doc_b.txt")
    print(f"DOC B upload time: {time_b: 3f}s")

    time_c = upload_document(DOC_C, "doc_c.txt")
    print(f"DOC C upload time: {time_b: 3f}s")

    assert time_b < time_a * 2, \
        f"second upload took too long ({time_b:.3f}s vs {time_a:.3f}s)"
    assert time_c < time_a * 2, \
        f"third upload took too long ({time_c:.3f}s vs {time_a:.3f}s)"
    
    print(f"upload times are consistent")

def test_incremental_indexing_retrieval():
    """testing whether all incrementally uploaded docs are searchable"""
    
    upload_document(DOC_A, "doc_a.txt")
    upload_document(DOC_B, "doc_b.txt")
    upload_document(DOC_C, "doc_c.txt")

    queries = [
        ("how many people were on the ship?", ["17"]),
        ("how many employees are in the warehouse?", ["89"]),
        ("how many machines does factory have?", ["52"])
    ]

    for query, expected_keywords in queries:
        response = client.post("/api/query", json={"query": query})
        assert response.status_code == 200

        data = response.json()  # << inside the loop, each query checked
        answer = data["answer"].lower()

        print(f"\nQuery: {query}")
        print(f"Answer: {data['answer']}")
        print(f"Sources: {data['sources']}")

        found_keyword = any(keyword.lower() in answer for keyword in expected_keywords)
        assert found_keyword, \
            f"none of {expected_keywords} found in answer - {data['answer']}"
    
    print("\n all incrementally uploaded documents are searchable")

def test_incremental_chunk_count():
    """testing that chunk counts increase with each upload"""

    upload_document(DOC_A, "doc_a.txt")
    chunks_after_a = len(app_module.chunks)
    print(f"chunks after DOC A: {chunks_after_a}")
    assert chunks_after_a > 0

    upload_document(DOC_B, "doc_b.txt")
    chunks_after_b = len(app_module.chunks)
    print(f"chunks after DOC B: {chunks_after_b}")
    assert chunks_after_b > chunks_after_a, "Chunk count did not increase"

    upload_document(DOC_C, "doc_c.txt")
    chunks_after_c = len(app_module.chunks)
    print(f"Chunks after doc C: {chunks_after_c}")
    assert chunks_after_c > chunks_after_b, "Chunk count did not increase"

    print(f"chunk counts increased correctly: {chunks_after_a} → {chunks_after_b} → {chunks_after_c}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])