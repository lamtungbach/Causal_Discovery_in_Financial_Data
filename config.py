"""
config.py — Tham số cấu hình toàn bộ dự án.
Chỉ sửa file này khi muốn thay đổi tham số.
"""

# ── Thời gian dữ liệu ─────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# ── Thị trường ────────────────────────────────────────────
TICKERS = {
    "SP500":   "^GSPC",
    "VNINDEX": None,
    "Gold":    "GC=F",
    "Oil":     "CL=F",
    "Bitcoin": "BTC-USD",
}

LABELS = {
    "SP500":   "S&P 500",
    "VNINDEX": "VN-Index",
    "Gold":    "Vàng (Gold)",
    "Oil":     "Dầu thô (WTI)",
    "Bitcoin": "Bitcoin",
}

NODE_NAMES = list(LABELS.values())

COLORS = {
    "SP500":   "#4FC3F7",
    "VNINDEX": "#81C784",
    "Gold":    "#FFD700",
    "Oil":     "#FFB74D",
    "Bitcoin": "#FF7043",
}

# ── Đường dẫn file thủ công ───────────────────────────────
MANUAL_FILES = {
    "VNINDEX": "data/raw/vnindex_manual.csv",
}

# ── Tham số xử lý dữ liệu ─────────────────────────────────
ROLLING_WINDOW = 21
FFILL_LIMIT    = 3

# ── File input cho các mô hình ────────────────────────────
NOTEARS_INPUT = "data/processed/volatility_stationary.csv"
GRANGER_INPUT = "data/processed/volatility_stationary.csv"
PC_INPUT      = "data/processed/volatility_stationary.csv"

# ── Tham số NOTEARS ───────────────────────────────────────
NOTEARS_LAMBDA = 0.1
NOTEARS_THRESH = 0.1

# ── Tham số PC Algorithm ──────────────────────────────────
PC_ALPHA = 0.1

# ── Tham số Granger Causality ─────────────────────────────
GRANGER_MAX_LAG = 5
GRANGER_ALPHA   = 0.05

# ── Đường dẫn thư mục ─────────────────────────────────────
PATH_RAW       = "data/raw"
PATH_PROCESSED = "data/processed"
PATH_RESULTS   = "data/results"
PATH_FIGURES   = "figures"
PATH_REPORTS   = "reports"

import pathlib
REPORTS_DIR = pathlib.Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR  = REPORTS_DIR / "tables"
LOGS_DIR    = REPORTS_DIR / "logs"

# ── Alias để tương thích các module cũ ───────────────────
NOTEARS_LAMBDA1   = NOTEARS_LAMBDA
NOTEARS_THRESHOLD = NOTEARS_THRESH
PC_CI_TEST        = "fisherz"
BOOTSTRAP_N       = 100
MARKET_COLORS     = COLORS
MARKET_LABELS     = LABELS
FIG_DPI           = 300

# ── Subperiods phân tích ──────────────────────────────────
SUBPERIODS = {
    "Pre-COVID"      : ("2018-01-01", "2020-02-28"),
    "COVID-19"       : ("2020-03-01", "2021-06-30"),
    "Post-COVID"     : ("2021-07-01", "2022-04-30"),
    "Crypto-Crash"   : ("2022-05-01", "2022-12-31"),
    "Banking-Crisis" : ("2023-01-01", "2023-06-30"),
    "Recovery"       : ("2023-07-01", "2024-12-31"),
}