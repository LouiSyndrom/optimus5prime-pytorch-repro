from pathlib import Path

# Set project file paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data/GSM3130435_egfp_unmod_1.csv.gz"
RESULTS_DIR = PROJECT_DIR / "results"

# Set data and split values
SEED = 42
SEQUENCE_LENGTH = 50
BASE_ORDER = "ATGC"
FRACTION_COLUMNS = [f"r{i}" for i in range(14)]
EXPECTED_ROWS = 326_033
HIGH_READ_SIZE = 20_000
TRAIN_SIZE = 240_000
VALIDATION_SIZE = 20_000

# Set CNN training values
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
AUXILIARY_LOSS_WEIGHT = 0.1

# Set baseline model values
RIDGE_PARAMS = {"alpha": 1000.0}
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 24,
    "min_samples_leaf": 20,
    "max_features": 1.0,
    "max_leaf_nodes": 10_000,
    "bootstrap": True,
    "max_samples": 0.8,
    "criterion": "squared_error",
}
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

# List supported models and saved files
MODEL_NAMES = ["ridge", "random_forest", "xgboost", "cnn"]
MODEL_FILES = {
    "ridge": "ridge.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
    "cnn": "best_model.pth",
}
