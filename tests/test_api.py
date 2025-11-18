import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import sys
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import api
import app as app_module

client = TestClient(api)

class SemanticEvaluator:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if SemanticEvaluator._model is None:
            SemanticEvaluator._model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = SemanticEvaluator._model
    
    def calculate_similarity(self, answer: str, expected: str) -> float:
        embeddings = self.model.encode([answer, expected])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def is_semantically_correct(self, answer: str, expected: str, threshold = 0.5) -> bool:
        return self.calculate_similarity(answer, expected) >= threshold

TEST_CONTENT = """
There were 17 people on the ship yesterday.
The captain's name was Jack.
The ship carried 150 livestock units.
Out of 150 livestock units - 67 cows, 22 chicken, 34 sheeps, 27 goats.
The Go programming language was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson in 2007.
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
        queries = [
            {
                "query": "how many livestock units were there on the ship yesterday?",
                "expected_keywords": ["150",],
                "should_not_contain": ["67", "22", "34", "27"],
                "expected_answer": "There were 150 livestock units on the ship"
            },
            {
                "query": "how many cows were on the ship?",
                "expected_keywords": ["67"],
                "should_not_contain": ["22", "34", "27", "150"],
                "expected_answer": "There were 67 cows on the ship"
            },
            {
                "query": "how many chicken were on the ship?",
                "expected_keywords": ["22"],
                "should_not_contain": ["67", "34", "27", "150"],
                "expected_answer": "There were 22 chicken on the ship"
            },
            {
                "query": "how many sheeps were on the ship?",
                "expected_keywords": ["34"],
                "should_not_contain": ["67", "22", "27", "150"],
                "expected_answer": "There were 34 sheep on the ship"
            },
            {
                "query": "how many goats were on the ship?",
                "expected_keywords": ["27"],
                "should_not_contain": ["67", "22", "34", "150"],
                "expected_answer": "There were 27 goats on the ship"
            },
            {
                "query": "who created the Go programming language?",
                "expected_keywords": ["Griesemer", "Pike", "Thompson"],
                "should_not_contain": ["Guido", "Linus"],
                "expected_answer": "Go was created by Robert Griesemer, Rob Pike, and Ken Thompson"
            }
        ]
        selected_query = random.choice(queries)
        query_response = client.post(
                "/api/query",
                json={"query": selected_query["query"]}
            )

        assert query_response.status_code == 200
        data = query_response.json()
        answer = data["answer"].lower()

        print("query:", selected_query["query"])
        print("answer:", data["answer"])
        print("sources:", data["sources"])

        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

        for keyword in selected_query["expected_keywords"]:
            assert keyword.lower() in answer, \
                f"Expected keyword '{keyword}' not found in answer: {data['answer']}"

        for forbidden in selected_query["should_not_contain"]:
            assert forbidden.lower() not in answer, \
                f"Answer contains incorrect info '{forbidden}': {data['answer']}"

        evaluator = SemanticEvaluator.get_instance()
        similarity = evaluator.calculate_similarity(data["answer"], selected_query["expected_answer"])
        print(f"semantic similarity: {similarity:.3f}")

        assert similarity >= 0.6, \
            f"answer semantically different from expected (similarity: {similarity:.3f})"

    finally:
        Path(temp_path).unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
