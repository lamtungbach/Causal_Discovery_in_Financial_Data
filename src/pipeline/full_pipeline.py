"""
src/pipeline/full_pipeline.py
==============================
FullPipeline — Steps B1 đến B8

Primary method  : NeuralGrangerCausality (cMLP + cLSTM)
Comparison      : PCAlgorithm, NOTEARS

B1. Load & preprocess data        → self.data_, self.variable_names_
B2. Stationarity validation       → ADF + KPSS
B3. NeuralGrangerCausality        → PRIMARY
B4. PCAlgorithm                   → comparison
B5. NOTEARS                       → comparison
B6. Evaluate (SHD, F1, Bootstrap) → self.metrics_
B7. Risk Scores + EWS             → self.risk_scores_
B8. Save & visualise
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.neural_granger import NeuralGrangerCausality
from src.models.base_model import CausalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Safe import helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_pc():
    try:
        from src.models.pc_algorithm import PCAlgorithm
        return PCAlgorithm
    except ImportError:
        logger.warning("PCAlgorithm chưa có — bỏ qua")
        return None

def _import_notears():
    try:
        from src.models.notears import NOTEARS
        return NOTEARS
    except ImportError:
        logger.warning("NOTEARS chưa có — bỏ qua")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(obj: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved → {path}")

def _save_adj_csv(adj: np.ndarray, names: list, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(adj, index=names, columns=names)
    df.to_csv(path)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FullPipeline
# ─────────────────────────────────────────────────────────────────────────────

class FullPipeline:
    """
    Orchestrates the full causal discovery pipeline.

    Parameters
    ----------
    config : dict  — loaded from run_all.py CONFIG
    """

    def __init__(self, config: dict):
        self.config = config

        # Output dir
        self.output_dir = Path(config.get("output_dir", "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data dirs
        data_cfg = config.get("data", {})
        self.processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
        self.raw_dir       = Path(data_cfg.get("raw_dir",       "data/raw"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.data_          : np.ndarray | None       = None
        self.variable_names_: list[str]  | None       = None
        self._vol_df        : pd.DataFrame | None     = None
        self.results_       : dict[str, CausalResult] = {}
        self.metrics_       : dict                    = {}
        self.risk_scores_   : dict                    = {}

    # ──────────────────────────────────────────────────────────────────────────
    # B1 — Load & Preprocess
    # ──────────────────────────────────────────────────────────────────────────

    def step_b1_load_data(self) -> np.ndarray:
        """
        Load prices → tính rolling volatility → chuẩn hoá.
        Dùng DataPreprocessor.run() để tự tìm prices file.
        """
        logger.info("=== B1: Load & Preprocess ===")

        from src.data.preprocessor import DataPreprocessor
        data_cfg = self.config.get("data", {})

        prep = DataPreprocessor(
            start         = data_cfg.get("start", "2018-01-01"),
            end           = data_cfg.get("end",   "2024-12-31"),
            window        = data_cfg.get("volatility_window", 21),
            raw_dir       = str(self.raw_dir),
            processed_dir = str(self.processed_dir),
        )

        X, vol, node_names = prep.run(standardize=True)

        self.data_           = X
        self.variable_names_ = node_names
        self._vol_df         = vol

        logger.info(f"Data: {X.shape} | Variables: {node_names}")
        print(f"\n   ✅ Data: shape={X.shape}, variables={node_names}")
        return X

    # ──────────────────────────────────────────────────────────────────────────
    # B2 — Stationarity
    # ──────────────────────────────────────────────────────────────────────────

    def step_b2_stationarity(self) -> dict:
        logger.info("=== B2: Stationarity (ADF + KPSS) ===")
        try:
            from src.data.validator import run_validation
            vol_path = self.processed_dir / "volatility_21d.csv"
            if vol_path.exists():
                vol = pd.read_csv(vol_path, index_col=0, parse_dates=True)
                desc, stat = run_validation(vol)
            else:
                logger.warning("Không tìm thấy volatility_21d.csv — bỏ qua stationarity")
                return {}
        except Exception as e:
            logger.warning(f"Stationarity test lỗi (không ảnh hưởng pipeline): {e}")
            return {}
        return {"stationarity": "done"}

    # ──────────────────────────────────────────────────────────────────────────
    # B3 — NeuralGrangerCausality (PRIMARY)
    # ──────────────────────────────────────────────────────────────────────────

    def step_b3_neural_granger(self) -> dict:
        if self.data_ is None:
            raise RuntimeError("Cần chạy step_b1_load_data() trước")

        logger.info("=== B3: NeuralGrangerCausality [PRIMARY] ===")
        cfg = self.config.get("neural_granger", {})

        from src.models.neural_granger import (
            NeuralGrangerConfig, NeuralGrangerCausality, compare_architectures
        )
        from src.models.base_model import CausalResult, ModelType
        import time

        base_config = NeuralGrangerConfig(
            lag          = cfg.get("lag",         1),
            hidden       = cfg.get("lstm_hidden", 32),
            lambda_group = cfg.get("lam",         0.01),
            lr           = cfg.get("lr",          1e-3),
            max_epochs   = cfg.get("max_epochs",  500),
            patience     = cfg.get("patience",    50),
            threshold    = cfg.get("threshold",   0.1),
            device       = cfg.get("device",      "cpu"),
            verbose      = False,
        )

        t0 = time.perf_counter()
        results = compare_architectures(self.data_, self.variable_names_, base_config)
        runtime = time.perf_counter() - t0

        adj_mlp  = results["cMLP"].adjacency_matrix
        adj_lstm = results["cLSTM"].adjacency_matrix
        adj_ens_strength = (adj_mlp + adj_lstm) / 2.0
        np.fill_diagonal(adj_ens_strength, 0)

        # Threshold trên strength trung bình
        adj_ens = (adj_ens_strength > base_config.threshold).astype(float)
        np.fill_diagonal(adj_ens, 0)

        res_mlp = CausalResult(
            method_name      = "NeuralGC_cMLP",
            adjacency_matrix = adj_mlp,
            binary_matrix    = (adj_mlp > base_config.threshold).astype(int),
            runtime_seconds  = runtime / 2,
            model_type       = ModelType.NEURAL_GRANGER_CMLP,
        )
        res_lstm = CausalResult(
            method_name      = "NeuralGC_cLSTM",
            adjacency_matrix = adj_lstm,
            binary_matrix    = (adj_lstm > base_config.threshold).astype(int),
            runtime_seconds  = runtime / 2,
            model_type       = ModelType.NEURAL_GRANGER_CLSTM,
        )
        res_ens = CausalResult(
            method_name      = "NeuralGC_Ensemble",
            adjacency_matrix = adj_ens,
            binary_matrix    = adj_ens.astype(int),
            runtime_seconds  = runtime,
            model_type       = ModelType.NEURAL_GRANGER_CLSTM,
        )

        self.results_["NeuralGC_cMLP"]     = res_mlp
        self.results_["NeuralGC_cLSTM"]    = res_lstm
        self.results_["NeuralGC_Ensemble"] = res_ens

        for key in ("NeuralGC_cMLP", "NeuralGC_cLSTM", "NeuralGC_Ensemble"):
            _save_adj_csv(
                self.results_[key].adjacency_matrix,
                self.variable_names_,
                self.output_dir / f"adj_{key}.csv",
            )

        print(f"   cMLP     edges : {int((adj_mlp > base_config.threshold).sum())}")
        print(f"   cLSTM    edges : {int((adj_lstm > base_config.threshold).sum())}")
        print(f"   Ensemble edges : {int(adj_ens.sum())}")
        print(f"   Runtime        : {runtime:.1f}s")
        return {"cMLP": res_mlp, "cLSTM": res_lstm, "Ensemble": res_ens}

    # ──────────────────────────────────────────────────────────────────────────
    # B4 — PC Algorithm
    # ──────────────────────────────────────────────────────────────────────────

    def step_b4_pc_algorithm(self) -> CausalResult | None:
        if self.data_ is None:
            raise RuntimeError("Cần chạy step_b1_load_data() trước")

        PCAlgorithm = _import_pc()
        if PCAlgorithm is None:
            return None

        logger.info("=== B4: PC Algorithm [COMPARISON] ===")
        cfg = self.config.get("pc", {})

        pc = PCAlgorithm(alpha=cfg.get("alpha", 0.05), node_names=self.variable_names_)
        result = pc.fit(self.data_)
        self.results_["PC"] = result

        _save_adj_csv(result.adjacency_matrix, self.variable_names_,
                      self.output_dir / "adj_PC.csv")
        print(f"\n   PC edges: {result.n_edges}")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # B5 — NOTEARS
    # ──────────────────────────────────────────────────────────────────────────

    def step_b5_notears(self) -> CausalResult | None:
        if self.data_ is None:
            raise RuntimeError("Cần chạy step_b1_load_data() trước")

        NOTEARS = _import_notears()
        if NOTEARS is None:
            return None

        logger.info("=== B5: NOTEARS [COMPARISON] ===")
        cfg = self.config.get("notears", {})

        nt = NOTEARS(
            lambda1   = cfg.get("lambda1",   0.1),
            threshold = cfg.get("threshold", 0.3),
            max_iter  = cfg.get("max_iter",  100),
            node_names = self.variable_names_
        )
        result = nt.fit(self.data_)
        self.results_["NOTEARS"] = result

        _save_adj_csv(result.adjacency_matrix, self.variable_names_,
                      self.output_dir / "adj_NOTEARS.csv")
        print(f"\n   NOTEARS edges: {result.n_edges}")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # B6 — Evaluate
    # ──────────────────────────────────────────────────────────────────────────

    def step_b6_evaluate(self) -> dict:
        if not self.results_:
            raise RuntimeError("Cần chạy step_b3_neural_granger() trước")

        logger.info("=== B6: Evaluate (SHD, Precision, Recall, F1) ===")

        # Pseudo ground truth = NeuralGC Ensemble
        gt_adj = self.results_["NeuralGC_Ensemble"].adjacency_matrix

        metrics = {}
        for name, result in self.results_.items():
            if name == "NeuralGC_Ensemble":
                continue
            m = self._compute_metrics(result.adjacency_matrix, gt_adj, name)
            metrics[name] = m
            print(f"\n   [{name}]  SHD={m['SHD']}  "
                  f"P={m['Precision']:.3f}  R={m['Recall']:.3f}  F1={m['F1']:.3f}")

        self.metrics_ = metrics
        _save_json(metrics, self.output_dir / "metrics.json")
        return metrics

    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray, name: str) -> dict:
        tp  = int(((pred == 1) & (gt == 1)).sum())
        fp  = int(((pred == 1) & (gt == 0)).sum())
        fn  = int(((pred == 0) & (gt == 1)).sum())
        shd = fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        return {
            "method"   : name,
            "TP"       : tp, "FP": fp, "FN": fn,
            "SHD"      : shd,
            "Precision": round(precision, 4),
            "Recall"   : round(recall,    4),
            "F1"       : round(f1,        4),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # B7 — Risk Scores + EWS
    # ──────────────────────────────────────────────────────────────────────────

    def step_b7_risk_and_ews(self) -> dict:
        logger.info("=== B7: Risk Scores & EWS ===")

        if "NeuralGC_Ensemble" not in self.results_:
            logger.warning("Chưa có NeuralGC_Ensemble — dùng zero matrix")
            p   = len(self.variable_names_) if self.variable_names_ else 5
            adj = np.zeros((p, p), dtype=int)
        else:
            adj = self.results_["NeuralGC_Ensemble"].adjacency_matrix

        names = self.variable_names_ or [f"X{i}" for i in range(adj.shape[0])]

        risk_scores = self._compute_risk_scores(adj, names)
        self.risk_scores_ = risk_scores
        _save_json(risk_scores, self.output_dir / "risk_scores.json")

        alerts = self._run_ews(adj, names)
        _save_json({"alerts": alerts}, self.output_dir / "ews_alerts.json")

        print(f"\n   Risk scores computed for: {list(risk_scores.keys())}")
        print(f"   EWS alerts: {len(alerts)} markets flagged")
        return {"risk_scores": risk_scores, "alerts": alerts}

    def _compute_risk_scores(self, adj: np.ndarray, names: list) -> dict:
        scores = {}
        for i, name in enumerate(names):
            out_deg = int(adj[i, :].sum()) - int(adj[i, i])
            in_deg  = int(adj[:, i].sum()) - int(adj[i, i])
            net     = out_deg - in_deg
            role    = "Transmitter" if net > 0 else ("Receiver" if net < 0 else "Neutral")
            scores[name] = {
                "transmitter_score": out_deg,
                "receiver_score"   : in_deg,
                "net_score"        : net,
                "role"             : role,
            }
        return scores

    def _run_ews(self, adj: np.ndarray, names: list) -> list:
        if self.data_ is None:
            return []
        cfg      = self.config.get("ews", {})
        z_thresh = cfg.get("threshold_z", 2.0)
        alerts   = []
        for j, name in enumerate(names):
            s    = self.data_[:, j]
            mean = s.mean()
            std  = s.std()
            if std == 0:
                continue
            z              = (s - mean) / std
            high_vol_idx   = np.where(np.abs(z) > z_thresh)[0]
            if len(high_vol_idx) > 0:
                alerts.append({
                    "market"         : name,
                    "n_alert_days"   : int(len(high_vol_idx)),
                    "max_z"          : float(np.abs(z).max()),
                    "first_alert_idx": int(high_vol_idx[0]),
                    "last_alert_idx" : int(high_vol_idx[-1]),
                })
        return alerts

    # ──────────────────────────────────────────────────────────────────────────
    # B8 — Save & Visualise
    # ──────────────────────────────────────────────────────────────────────────

    def step_b8_save_and_visualise(self):
        logger.info("=== B8: Save & Visualise ===")
        methods_dict = {}
        for k, v in self.results_.items():
            methods_dict[k] = {
                "method_name": v.method_name,
                "model_type": v.model_type.value if hasattr(v.model_type, 'value') else str(v.model_type),
                "n_edges": v.n_edges,
                "runtime_seconds": v.runtime_seconds,
            }
        
        summary = {
            "variable_names": self.variable_names_,
            "n_methods"     : len(self.results_),
            "methods"       : methods_dict,
            "metrics"       : self.metrics_,
            "risk_scores"   : self.risk_scores_,
        }
        _save_json(summary, self.output_dir / "full_results.json")

        print(f"\n   📁 Kết quả lưu tại: {self.output_dir}/")
        for k in self.results_:
            print(f"      adj_{k}.csv  ({self.results_[k].n_edges} edges)")
        if self.metrics_:
            print(f"\n   📋 Evaluation (vs NeuralGC Ensemble):")
            for k, m in self.metrics_.items():
                print(f"      {k:<22}: SHD={m['SHD']:>3}  F1={m['F1']:.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Chạy toàn bộ B1 → B8."""
        self.step_b1_load_data()
        self.step_b2_stationarity()
        self.step_b3_neural_granger()
        self.step_b4_pc_algorithm()
        self.step_b5_notears()
        self.step_b6_evaluate()
        self.step_b7_risk_and_ews()
        self.step_b8_save_and_visualise()
        return {
            "results"    : self.results_,
            "metrics"    : self.metrics_,
            "risk_scores": self.risk_scores_,
        }