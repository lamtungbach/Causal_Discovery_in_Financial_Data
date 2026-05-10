"""
src/utils/io_utils.py
Save / Load kết quả nghiên cứu — CausalResult, DataFrames, figures, JSON.

Đặt file tại: src/utils/io_utils.py

Tính năng:
  - Lưu / load CausalResult (pickle + JSON summary)
  - Export bảng so sánh ra CSV, Excel, LaTeX
  - Snapshot toàn bộ pipeline (để tái sử dụng không cần chạy lại)
  - Kiểm tra integrity (hash) khi load

Sử dụng
-------
>>> io = ResultIO(base_dir="reports")
>>> io.save_result("NOTEARS", notears_result)
>>> io.save_comparison(comparison_df)
>>> io.snapshot(pipeline.results_, comparison_df)

>>> # Load lại sau này
>>> notears_result = io.load_result("NOTEARS")
>>> comparison_df  = io.load_comparison()
"""

import os
import sys
import json
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

try:
    from config import TABLES_DIR, FIGURES_DIR
except ImportError:
    TABLES_DIR  = Path("reports/tables")
    FIGURES_DIR = Path("reports/figures")

logger = logging.getLogger("KLTN.src.utils.io_utils")


# ─────────────────────────────────────────────────────────────────────────────

class ResultIO:
    """
    Quản lý toàn bộ I/O cho kết quả nghiên cứu.

    Parameters
    ----------
    base_dir : str  thư mục gốc. Mặc định "reports/".

    Cấu trúc thư mục được tạo tự động:
        reports/
        ├── tables/          CSV, Excel, LaTeX
        ├── figures/         PNG plots
        ├── checkpoints/     Pickle snapshots
        └── logs/            Log files
    """

    def __init__(self, base_dir: str = "reports"):
        self.base_dir    = Path(base_dir)
        self.tables_dir  = self.base_dir / "tables"
        self.figures_dir = self.base_dir / "figures"
        self.ckpt_dir    = self.base_dir / "checkpoints"

        for d in [self.tables_dir, self.figures_dir, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # CausalResult — save / load
    # =========================================================================

    def save_result(
        self,
        method_name: str,
        result     : object,
        overwrite  : bool = True,
    ) -> Path:
        """
        Lưu CausalResult ra 2 file:
          - .pkl   (pickle, đầy đủ để load lại)
          - .json  (summary dạng người đọc được)

        Parameters
        ----------
        method_name : str    "NOTEARS" | "Granger" | "PC Algorithm"
        result      : CausalResult
        overwrite   : bool   ghi đè nếu đã tồn tại

        Returns
        -------
        Path  đường dẫn file .pkl
        """
        safe_name = _safe_filename(method_name)
        pkl_path  = self.ckpt_dir / f"result_{safe_name}.pkl"
        json_path = self.ckpt_dir / f"result_{safe_name}.json"

        if pkl_path.exists() and not overwrite:
            logger.warning(f"  {pkl_path} đã tồn tại, bỏ qua (overwrite=False)")
            return pkl_path

        # Lưu pickle
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Lưu JSON summary (để xem nhanh, không cần Python)
        summary = self._result_to_json(method_name, result)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã lưu {method_name}: {pkl_path.name} + {json_path.name}")
        return pkl_path

    def load_result(self, method_name: str) -> object:
        """
        Load CausalResult từ file pickle.

        Parameters
        ----------
        method_name : str   "NOTEARS" | "Granger" | "PC Algorithm"

        Returns
        -------
        CausalResult

        Raises
        ------
        FileNotFoundError  nếu chưa save trước đó
        """
        safe_name = _safe_filename(method_name)
        pkl_path  = self.ckpt_dir / f"result_{safe_name}.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint cho '{method_name}'. "
                f"Chạy pipeline và save_result() trước."
            )

        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        logger.info(f"Đã load {method_name} ← {pkl_path.name}")
        return result

    def save_all_results(self, results: Dict[str, object]) -> None:
        """Lưu tất cả CausalResult trong dict."""
        for name, result in results.items():
            self.save_result(name, result)

    def load_all_results(self, method_names: list) -> Dict[str, object]:
        """Load nhiều CausalResult cùng lúc."""
        return {name: self.load_result(name) for name in method_names}

    # =========================================================================
    # DataFrames — CSV / Excel / LaTeX
    # =========================================================================

    def save_comparison(
        self,
        df         : pd.DataFrame,
        filename   : str = "method_comparison",
        formats    : tuple = ("csv", "excel", "latex"),
    ) -> None:
        """
        Lưu bảng so sánh 3 phương pháp ra nhiều định dạng.

        Parameters
        ----------
        df       : pd.DataFrame   output từ MethodComparator.comparison_table()
        filename : str            tên file (không có đuôi)
        formats  : tuple          ("csv", "excel", "latex") — chọn định dạng cần lưu
        """
        if "csv" in formats:
            p = self.tables_dir / f"{filename}.csv"
            df.to_csv(p)
            logger.info(f"CSV   → {p}")

        if "excel" in formats:
            p = self.tables_dir / f"{filename}.xlsx"
            try:
                with pd.ExcelWriter(p, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Comparison")
                    # Auto-fit column width
                    ws = writer.sheets["Comparison"]
                    for col in ws.columns:
                        max_len = max(len(str(cell.value or "")) for cell in col)
                        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)
                logger.info(f"Excel → {p}")
            except ImportError:
                logger.warning("pip install openpyxl để lưu Excel")

        if "latex" in formats:
            p = self.tables_dir / f"{filename}.tex"
            latex_str = df.to_latex(
                escape    = False,
                bold_rows = True,
                caption   = "So sánh hiệu suất 3 phương pháp Causal Discovery",
                label     = "tab:method_comparison",
            )
            p.write_text(latex_str, encoding="utf-8")
            logger.info(f"LaTeX → {p}")

    def load_comparison(self, filename: str = "method_comparison") -> pd.DataFrame:
        """Load bảng so sánh từ CSV."""
        p = self.tables_dir / f"{filename}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy {p}")
        df = pd.read_csv(p, index_col=0)
        logger.info(f"Đã load comparison ← {p}")
        return df

    def save_csv(
        self,
        df      : pd.DataFrame,
        filename: str,
        index   : bool = True,
    ) -> Path:
        """Lưu bất kỳ DataFrame ra CSV."""
        p = self.tables_dir / (filename if filename.endswith(".csv") else f"{filename}.csv")
        df.to_csv(p, index=index)
        logger.info(f"CSV → {p}")
        return p

    # =========================================================================
    # Snapshot toàn bộ pipeline
    # =========================================================================

    def snapshot(
        self,
        results       : Dict[str, object],
        comparison_df : Optional[pd.DataFrame] = None,
        vol_df        : Optional[pd.DataFrame] = None,
        metadata      : Optional[dict] = None,
    ) -> Path:
        """
        Lưu snapshot toàn bộ pipeline ra 1 file pickle duy nhất.
        Dùng để tái sử dụng kết quả mà không cần chạy lại pipeline (~5-10 phút).

        Parameters
        ----------
        results       : dict             {"NOTEARS": result, "Granger": result, ...}
        comparison_df : pd.DataFrame     bảng so sánh
        vol_df        : pd.DataFrame     volatility data
        metadata      : dict             thông tin tùy chọn (config, timestamp, ...)

        Returns
        -------
        Path  đường dẫn file snapshot

        Example
        -------
        >>> io.snapshot(pipeline.results_, comparison, vol_df)
        # Lần sau chỉ cần:
        >>> data = io.load_snapshot()
        >>> results = data["results"]
        """
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.ckpt_dir / f"snapshot_{ts}.pkl"

        payload = {
            "timestamp"  : ts,
            "results"    : results,
            "comparison" : comparison_df,
            "vol_df"     : vol_df,
            "metadata"   : metadata or {},
        }

        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Ghi hash để verify sau này
        file_hash = _file_hash(out_path)
        hash_path = out_path.with_suffix(".sha256")
        hash_path.write_text(file_hash)

        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(
            f"Snapshot → {out_path.name} "
            f"({size_mb:.1f} MB | sha256={file_hash[:12]}...)"
        )

        # Tạo symlink "latest" để dễ load
        latest = self.ckpt_dir / "snapshot_latest.pkl"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(out_path.name)
        except (OSError, NotImplementedError):
            # Windows không hỗ trợ symlink → copy thay thế
            import shutil
            shutil.copy2(out_path, latest)

        return out_path

    def load_snapshot(
        self,
        path    : Optional[str] = None,
        verify  : bool = True,
    ) -> dict:
        """
        Load snapshot.

        Parameters
        ----------
        path   : str, optional  đường dẫn cụ thể. Nếu None → load latest.
        verify : bool           kiểm tra hash SHA256.

        Returns
        -------
        dict  {"results", "comparison", "vol_df", "metadata", "timestamp"}
        """
        if path is None:
            snap_path = self.ckpt_dir / "snapshot_latest.pkl"
        else:
            snap_path = Path(path)

        if not snap_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy snapshot tại {snap_path}. "
                "Chạy pipeline.run() và io.snapshot() trước."
            )

        # Verify hash
        if verify:
            hash_path = snap_path.with_suffix(".sha256")
            if hash_path.exists():
                expected = hash_path.read_text().strip()
                actual   = _file_hash(snap_path)
                if expected != actual:
                    raise ValueError(
                        f"Hash không khớp! File có thể bị hỏng.\n"
                        f"Expected: {expected}\nActual  : {actual}"
                    )
                logger.debug(f"Hash OK: {actual[:12]}...")
            else:
                logger.warning("Không tìm thấy file .sha256 — bỏ qua verify")

        with open(snap_path, "rb") as f:
            data = pickle.load(f)

        logger.info(
            f"Đã load snapshot ← {snap_path.name} "
            f"(created: {data.get('timestamp', 'unknown')})"
        )
        return data

    def list_snapshots(self) -> pd.DataFrame:
        """Liệt kê tất cả snapshot đã lưu."""
        snaps = sorted(self.ckpt_dir.glob("snapshot_2*.pkl"))
        rows  = []
        for p in snaps:
            size_mb = p.stat().st_size / 1024 / 1024
            rows.append({
                "File"      : p.name,
                "Size (MB)" : round(size_mb, 2),
                "Modified"  : datetime.fromtimestamp(p.stat().st_mtime)
                               .strftime("%Y-%m-%d %H:%M:%S"),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["File", "Size (MB)", "Modified"]
        )

    # =========================================================================
    # Report generation
    # =========================================================================

    def export_full_report(
        self,
        results      : Dict[str, object],
        comparison_df: pd.DataFrame,
        config_dict  : Optional[dict] = None,
    ) -> Path:
        """
        Tạo file Excel báo cáo tổng hợp với nhiều sheet.

        Sheet 1 — Method Comparison (bảng 4.5)
        Sheet 2 — NOTEARS Edges (danh sách cạnh + trọng số)
        Sheet 3 — Degree Table   (in/out-degree từng thị trường)
        Sheet 4 — Config         (hyperparameters)

        Parameters
        ----------
        results       : dict
        comparison_df : pd.DataFrame
        config_dict   : dict, optional

        Returns
        -------
        Path  đường dẫn file .xlsx
        """
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            raise ImportError("pip install openpyxl")

        out_path = self.tables_dir / "full_report.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

            # Sheet 1: Method Comparison
            comparison_df.to_excel(
                writer, sheet_name="Method Comparison", index=True
            )

            # Sheet 2: NOTEARS edges
            if "NOTEARS" in results:
                edges = results["NOTEARS"].edge_list()
                if edges:
                    pd.DataFrame(edges).to_excel(
                        writer, sheet_name="NOTEARS Edges", index=False
                    )

            # Sheet 3: Degree tables
            for method_name, result in results.items():
                safe = _safe_filename(method_name)[:20]
                try:
                    result.degree_table().to_excel(
                        writer, sheet_name=f"Degree_{safe}"
                    )
                except Exception:
                    pass

            # Sheet 4: Config
            if config_dict:
                pd.DataFrame(
                    list(config_dict.items()),
                    columns=["Parameter", "Value"]
                ).to_excel(writer, sheet_name="Config", index=False)

        logger.info(f"Full report → {out_path}")
        return out_path

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _result_to_json(method_name: str, result: object) -> dict:
        """Chuyển CausalResult thành dict JSON-serializable."""
        return {
            "method"           : method_name,
            "n_edges"          : result.n_edges,
            "runtime_seconds"  : round(result.runtime_seconds, 4),
            "node_names"       : result.node_names,
            "hub_nodes"        : result.hub_nodes(top_k=3),
            "sink_nodes"       : result.sink_nodes(top_k=3),
            "edge_list"        : result.edge_list(),
            "binary_matrix"    : result.binary_matrix.tolist(),
            "metadata"         : result.metadata,
            "saved_at"         : datetime.now().isoformat(),
        }


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Chuyển tên phương pháp thành tên file hợp lệ."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "_")
    )


def _file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Tính SHA256 hash của file để verify integrity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()