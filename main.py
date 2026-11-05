from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from utils.config import Config


if __name__ == "__main__":
    if not Config.DATA_FILE.exists():
        preprocess()
    train()
    evaluate()