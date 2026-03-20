import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))    

from src.inference import SearchEngine
from utils.config import Config

if __name__ == "__main__":
    engine = SearchEngine()
    print("Type an emotion query. Type 'exit' to quit.")

    while True:
        query = input("Enter emotion: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break

        if query == "":
            print("Please enter a non-empty query.")
            continue

        results = engine.search(query, top_k=Config.SEARCH_TOP_K)

        print("Top matches:")
        for r in results:
            print(r)