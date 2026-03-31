import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))    

from src.inference import SearchEngine
from utils.config import Config


TEST_QUERIES = [
    "happy",
    "sad",
    "peaceful",
]


if __name__ == "__main__":
    start = time.time()
    engine = SearchEngine()
    print(f"Search engine ready in {time.time() - start:.2f}s")

    for query in TEST_QUERIES:
        q_start = time.time()
        results = engine.search(query, top_k=Config.SEARCH_TOP_K)

        print(f"\nQuery: {query} | time={time.time() - q_start:.2f}s")
        print("Top matches:")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result}")