"""
src/utils/logger.py
Chuẩn hóa logging cho toàn bộ project KLTN.

Đặt file tại: src/utils/logger.py

Tính năng:
  - Ghi đồng thời ra console (màu) và file (.log)
  - Tự động tạo tên file log theo timestamp
  - Hỗ trợ lấy logger con theo module name
  - Tích hợp progress bar (tqdm) nếu cần

Sử dụng
-------
# Khởi tạo 1 lần ở entry point (run_all.py / __main__)
>>> from src.utils.logger import setup_logger
>>> setup_logger("KLTN", level=logging.INFO)

# Lấy logger trong từng module
>>> from src.utils.logger import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Bắt đầu xử lý dữ liệu...")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── ANSI color codes cho console (tắt trên Windows nếu không hỗ trợ) ────────
_RESET  = "\033[0m"
_COLORS = {
    "DEBUG"   : "\033[36m",    # Cyan
    "INFO"    : "\033[32m",    # Green
    "WARNING" : "\033[33m",    # Yellow
    "ERROR"   : "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
}

# Kiểm tra terminal có hỗ trợ màu không
_USE_COLOR = sys.stdout.isatty()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Formatter với màu sắc
# ─────────────────────────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """Formatter thêm ANSI color cho levelname trên console."""

    FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if _USE_COLOR:
            color = _COLORS.get(record.levelname, "")
            level_colored = f"{color}{record.levelname:<8}{_RESET}"
            formatted = formatted.replace(record.levelname.ljust(8), level_colored, 1)
        return formatted

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


class _PlainFormatter(logging.Formatter):
    """Formatter thuần text cho file log (không có ANSI codes)."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(
    name      : str  = "KLTN",
    log_dir   : str  = "reports/logs",
    level     : int  = logging.INFO,
    console   : bool = True,
    file_log  : bool = True,
    propagate : bool = False,
) -> logging.Logger:
    """
    Khởi tạo root logger cho project.
    Gọi 1 lần duy nhất ở entry point (run_all.py).

    Parameters
    ----------
    name      : str   tên root logger (mặc định "KLTN")
    log_dir   : str   thư mục lưu file .log
    level     : int   logging.DEBUG | INFO | WARNING | ERROR
    console   : bool  in ra stdout không
    file_log  : bool  ghi ra file không
    propagate : bool  cho phép propagate lên root logger Python không

    Returns
    -------
    logging.Logger

    Example
    -------
    >>> logger = setup_logger("KLTN", level=logging.DEBUG)
    >>> logger.info("Pipeline bắt đầu")
    2024-01-15 09:30:00 | INFO     | KLTN                      | Pipeline bắt đầu
    """
    logger = logging.getLogger(name)

    # Tránh thêm handler trùng nếu gọi nhiều lần
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)   # Root level = DEBUG, handler tự filter
    logger.propagate = propagate

    # ── Console handler ───────────────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_ColorFormatter())
        logger.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────────
    if file_log:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"kltn_{ts}.log"

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)    # File lưu tất cả kể cả DEBUG
        fh.setFormatter(_PlainFormatter())
        logger.addHandler(fh)

        # Ghi header vào file log
        logger.debug("=" * 70)
        logger.debug(f"LOG FILE: {log_file.absolute()}")
        logger.debug(f"START   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.debug("=" * 70)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger con theo tên module.
    Logger con tự động kế thừa handlers từ root logger "KLTN".

    Parameters
    ----------
    name : str  thường dùng __name__ để tự động lấy tên module

    Returns
    -------
    logging.Logger

    Example
    -------
    # Trong src/models/notears.py:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Bắt đầu NOTEARS optimization...")
    2024-01-15 09:30:01 | INFO     | src.models.notears        | Bắt đầu NOTEARS optimization...
    """
    # Đảm bảo logger con luôn có prefix "KLTN."
    # nếu name bắt đầu bằng "src." thì đổi thành "KLTN.src."
    if not name.startswith("KLTN"):
        child_name = f"KLTN.{name}"
    else:
        child_name = name
    return logging.getLogger(child_name)


# ─────────────────────────────────────────────────────────────────────────────
# Context manager: timed block
# ─────────────────────────────────────────────────────────────────────────────

class LogTimer:
    """
    Context manager đo thời gian và tự động log.

    Sử dụng
    -------
    >>> with LogTimer("Chạy NOTEARS"):
    ...     result = model.fit(X)
    # Output:
    # INFO | ▶ Bắt đầu: Chạy NOTEARS
    # INFO | ✓ Hoàn thành: Chạy NOTEARS — 12.34s
    """

    def __init__(
        self,
        label     : str,
        logger_obj: Optional[logging.Logger] = None,
        level     : int = logging.INFO,
    ):
        import time
        self.label      = label
        self.logger_obj = logger_obj or logging.getLogger("KLTN")
        self.level      = level
        self._time      = time

    def __enter__(self):
        self._start = self._time.perf_counter()
        self.logger_obj.log(self.level, f"▶ Bắt đầu: {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self._time.perf_counter() - self._start
        if exc_type is None:
            self.logger_obj.log(
                self.level,
                f"✓ Hoàn thành: {self.label} — {elapsed:.2f}s"
            )
        else:
            self.logger_obj.error(
                f"✗ Lỗi trong: {self.label} — {elapsed:.2f}s | {exc_val}"
            )
        return False   # Không suppress exception