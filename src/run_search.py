from src.inference import SearchEngine

if __name__ == "__main__":
    engine = SearchEngine()

    while True:
        query = input("Enter emotion: ")
        results = engine.search(query)

        print("Top matches:")
        for r in results:
            print(r)