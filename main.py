if __name__ == "__main__":
    print("Startup: importing configuration...", flush=True)
    from utils.config import Config

    print(
        f"Startup: Kaggle={Config.IS_KAGGLE} | "
        f"CUDA available={__import__('torch').cuda.is_available()}",
        flush=True,
    )
    from src.preprocess import preprocess
    from src.train import train
    from src.evaluate import evaluate

    print("Startup: imports complete.", flush=True)
    if not Config.DATA_FILE.exists():
        print("Startup: dataset manifest missing; preprocessing...", flush=True)
        preprocess()
    else:
        print(f"Startup: using dataset manifest {Config.DATA_FILE}", flush=True)
    print("Startup: entering training...", flush=True)
    train()
    print("Training finished; entering evaluation...", flush=True)
    evaluate()