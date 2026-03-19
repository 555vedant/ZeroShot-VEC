import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)