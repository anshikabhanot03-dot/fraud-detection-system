from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_xgboost.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FALSE_POSITIVE_COST = 25.0
FALSE_NEGATIVE_COST = 2500.0

