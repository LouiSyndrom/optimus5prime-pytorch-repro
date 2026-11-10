import argparse

from src.config import DATA_FILE, MODEL_NAMES, RESULTS_DIR
from src.data import create_data_splits, load_data


# Read the command line options
def parse_args():
    parser = argparse.ArgumentParser(description="Run one model in an isolated process")
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("model", choices=MODEL_NAMES)
    return parser.parse_args()


# Run one model in its own process
def main():
    args = parse_args()
    # Recreate the shared data splits
    data = load_data(DATA_FILE)
    splits = create_data_splits(data, RESULTS_DIR / "split_indexes.npz")

    # Run the requested action
    if args.action == "train":
        from src.training import train_models

        train_models([args.model], splits, RESULTS_DIR)
    else:
        from src.evaluation import evaluate_saved_model

        evaluate_saved_model(args.model, splits, RESULTS_DIR)


# Start the command line runner
if __name__ == "__main__":
    main()
