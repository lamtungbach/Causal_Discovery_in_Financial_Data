"""
Cấu hình chung cho khóa luận AI-first.
Chạy mọi script từ thư mục gốc project:
    python scripts/01_validate_data.py
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
CAMERA_READY_DIR = RESULTS_DIR / "camera_ready"
CAMERA_TABLE_DIR = CAMERA_READY_DIR / "tables"
CAMERA_FIGURE_DIR = CAMERA_READY_DIR / "figures"
CAMERA_MODEL_DIR = CAMERA_READY_DIR / "models"
CAMERA_DATA_DIR = CAMERA_READY_DIR / "data"

for _d in [
    PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR,
    CAMERA_READY_DIR, CAMERA_TABLE_DIR, CAMERA_FIGURE_DIR,
    CAMERA_MODEL_DIR, CAMERA_DATA_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

PRICE_FILE = PROCESSED_DIR / "master_price.csv"
RETURN_FILE = PROCESSED_DIR / "master_return.csv"
FEATURE_FILE = PROCESSED_DIR / "dataset_features.csv"

ASSETS = {
    "vnindex": {"price": "close_vnindex", "return": "return_vnindex", "label": "VNI"},
    "btc":     {"price": "close_btc",     "return": "return_btc",     "label": "BTC"},
    "oil":     {"price": "close_oil",     "return": "return_oil",     "label": "OIL"},
    "gold":    {"price": "close_gold",    "return": "return_gold",    "label": "GOLD"},
    "sp500":   {"price": "close_sp500",   "return": "return_sp500",   "label": "SP500"},
    "sse":     {"price": "close_sse",     "return": "return_sse",     "label": "SSE"},
    "vix":     {"price": "close_vix",     "return": "return_vix",     "label": "VIX"},
}
RETURN_COLS = [v["return"] for v in ASSETS.values()]
PRICE_COLS = [v["price"] for v in ASSETS.values()]
LABELS = {v["return"]: v["label"] for v in ASSETS.values()}
TARGET_COL = "return_vnindex"

PERIODS = {
    "full":       ("2015-01-01", "2025-12-31"),
    "pre_covid":  ("2015-01-01", "2019-12-31"),
    "covid":      ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2025-12-31"),
}

EVENT_WINDOWS = {
    "sse_bubble_2015": ("2015-06-01", "2015-09-30"),
    "covid_crash_2020": ("2020-02-01", "2020-05-31"),
    "btc_crash_2021": ("2021-04-01", "2021-07-31"),
    "fed_hike_2022": ("2022-03-01", "2022-12-31"),
}

# Wavelet/MODWT bands. SWT/MODWT level j xấp xỉ horizon [2^j, 2^(j+1)] ngày.
WAVELET = "db4"
MAX_LEVEL = 6
BAND_LEVELS = {
    "short": [1, 2, 3],     # 2-16 ngày
    "medium": [4, 5],       # 16-64 ngày
    "long": [6],            # 64-128+ ngày
}

# Neural model defaults
DEFAULT_LAGS = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

# Camera-ready experiment contract. These values are deliberately separate
# from the lightweight thesis defaults above so old outputs are never silently
# overwritten by the revised experiments.
CAMERA_LAGS = 20
CAMERA_TRAIN_RATIO = 0.70
CAMERA_VAL_RATIO = 0.15
CAMERA_NGC_HIDDEN = (64, 64)
CAMERA_LSTM_HIDDEN = 64
CAMERA_NGC_LAMBDA = 5e-4
CAMERA_GRAPH_QUANTILE = 0.70
CAMERA_GBRI_ALPHA = 0.50
CAMERA_VOL_WINDOW = 20
CAMERA_STABILITY_SEEDS = tuple(range(42, 62))
CAMERA_STABILITY_THRESHOLD = 0.80
