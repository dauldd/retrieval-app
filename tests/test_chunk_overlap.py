import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import api
import app as app_module

client = TestClient(api)

LONG_DOC = """
The ancient library contained thousands of manuscripts from civilizations across the globe.
Scholars traveled from distant lands to study the precious texts housed within its towering walls.
Among the collection were works on astronomy, mathematics, philosophy, and natural sciences.
The head librarian, Marcus Aurelius Septimus, had spent forty-three years cataloging every single volume.
He knew the location of each manuscript by heart and could recite passages from memory.
The library's most prized possession was a first edition medical treatise written in 1247 AD.
This rare document described surgical techniques that were centuries ahead of their time.
The building itself was constructed from limestone blocks quarried from the nearby mountains.
Its architecture featured intricate geometric patterns and elaborate carved columns.
During the summer months, the library attracted over five thousand visitors annually.
Many came specifically to view the ancient star charts painted on the ceiling of the main hall.
These celestial maps depicted constellations observed by astronomers over eight hundred years ago.
The preservation team worked tirelessly to protect the fragile documents from humidity and decay.
They employed specialized techniques involving temperature control and careful handling procedures.
Each manuscript was stored in custom-made cases designed to prevent deterioration.
The library also maintained a detailed digital archive of all texts for future generations.
Researchers could access high-resolution scans of any document through the online portal.
This digital initiative ensured that knowledge would survive even if the physical copies were lost.
The institution received funding from both government sources and private philanthropic organizations.
Annual donations helped support the ongoing conservation efforts and educational programs.
Students from the local university frequently interned at the library during their academic studies.
They gained valuable experience in archival science and historical research methodologies.
""".strip()

@pytest.fixture(autouse=True)
def reset_app_state():
    """resetting app before and after test"""
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

def test_chunk_overlap():
    """testing that consecutive chunks have the expected 100-character overlap"""
   
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(LONG_DOC)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": ("library.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200

        chunks = app_module.chunks
        print(f"total chunks created: {len(chunks)}")

        assert len(chunks) > 1, "document should be split into multiple chunks"
        overlaps_found = 0

        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].page_content
            chunk2_text = chunks[i + 1].page_content

            print(f"\n--- chunk {i} vs chunk {i+1} ---")
            print(f"chunk {i} length: {len(chunk1_text)}")
            print(f"chunk {i+1} length: {len(chunk2_text)}")
            print(f"chunk {i} end (last 100 chars): ...{chunk1_text[-100:]}")
            print(f"chunk {i+1} start (first 100 chars): {chunk2_text[:100]}...")

            overlap_found = False

            for overlap_size in  range(50, min(151, len(chunk1_text), len(chunk2_text))):
                chunk1_end = chunk1_text[-overlap_size:]
                if chunk1_end in chunk2_text[:150]:
                    print(f"found overlap - {overlap_size} (characters)")
                    print(f"overlapping text - '{chunk1_end}'")
                    overlap_found = True
                    overlaps_found += 1

                    assert 50 <= overlap_size <= 150, \
                        f"overlap size {overlap_size} is outside the range"
                    break

            if not overlap_found:
                    print("no overlap found between consecutive chunks!")

        print(f"\n✓ Found {overlaps_found} overlaps out of {len(chunks)-1} consecutive chunk pairs")

    finally:
        Path(temp_path).unlink()

def test_no_information_loss():
    """testing whether all the information from the original document appears in the chunks"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(LONG_DOC)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": ("library.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200

        chunks = app_module.chunks

        all_chunk_text = " ".join([chunk.page_content for chunk in chunks])

        key_phrases = [
            "Marcus Aurelius Septimus",
            "forty-three years",
            "first edition medical treatise",
            "1247 AD",
            "five thousand visitors",
            "eight hundred years ago",
            "temperature control",
            "digital archive"
        ]

        print(f"Checking {len(key_phrases)} key phrases...")

        for phrase in key_phrases:
            found = phrase in all_chunk_text
            print(f"'{phrase}': {'found' if found else 'missing'}")

            assert found, f"key phrase '{phrase}' was lost during chunking!"

        print(f"\nall key phrases found - no information loss detected")

    finally:
        Path(temp_path).unlink()

def test_chunk_size_constraints(): # chars - characters
    """testing that chunk respect the expected size constraints (800 chars with 100 overlap)"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(LONG_DOC)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": ("library.txt", f, "text/plain")}
            )

        assert response.status_code == 200

        chunks = app_module.chunks

        print(f"analyzing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk.page_content)
            print(f"chunk{i}: {chunk_len} chars")

            assert 100 <= chunk_len <= 900, \
                f"chunk {i} has {chunk_len} chars!!"
        
        avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"\naverage chunk size: {avg_size:.1f} chars")
        print(f"expected ~ 800 chars")

        assert 400 <= avg_size <= 900, \
            f"average chunk size {avg_size:.1f} is far from expected 800 chars"
        
        print("chunk sizes are within expected constraints")

    finally:
        Path(temp_path).unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])