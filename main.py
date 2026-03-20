from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate


if __name__ == "__main__":
    preprocess()
    train()
    evaluate()