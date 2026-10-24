"""Read-only dashboard for the authoritative camera-ready outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CAMERA_FIGURE_DIR, CAMERA_TABLE_DIR  # noqa: E402


st.set_page_config(page_title="KSE 2026 Camera-ready Results", layout="wide")
st.title("KSE 2026 Camera-ready Results")
st.caption(
    "Dashboard chỉ đọc các kết quả trong results/camera_ready; "
    "các bảng và hình của pipeline cũ không được sử dụng."
)

metrics_path = CAMERA_TABLE_DIR / "forecast_metrics.csv"
if not metrics_path.exists():
    st.error("Chưa có kết quả camera-ready. Hãy chạy: python run_pipeline.py")
    st.stop()

overview_tab, graph_tab, stability_tab, robustness_tab, risk_tab = st.tabs(
    ["Forecast", "Graph", "Stability", "Robustness", "GBRI"]
)

with overview_tab:
    metrics = pd.read_csv(metrics_path)
    st.subheader("Out-of-sample forecast metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    metric_long = metrics.melt(id_vars="model", var_name="metric", value_name="value")
    st.bar_chart(metric_long, x="metric", y="value", color="model", stack=False)
    inference_path = CAMERA_TABLE_DIR / "forecast_inference.json"
    if inference_path.exists():
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        st.subheader("Forecast inference")
        st.json(inference, expanded=False)

with graph_tab:
    node_path = CAMERA_TABLE_DIR / "node_roles_prediction_weights.csv"
    if node_path.exists():
        nodes = pd.read_csv(node_path)
        st.subheader("Prediction-ablation graph roles (seed 42)")
        st.dataframe(nodes, use_container_width=True, hide_index=True)
        st.caption("Net strength: positive = transmitter, negative = receiver")
        st.bar_chart(nodes, x="node", y="net_strength", color="role", stack=False)
    graph_path = CAMERA_FIGURE_DIR / "prediction_weighted_ngc_graph.png"
    if graph_path.exists():
        st.image(str(graph_path), caption="Seed-42 retained NGC graph", use_container_width=True)

with stability_tab:
    edge_path = CAMERA_TABLE_DIR / "edge_stability_20seeds.csv"
    role_path = CAMERA_TABLE_DIR / "node_role_stability_20seeds.csv"
    if edge_path.exists():
        edges = pd.read_csv(edge_path)
        st.subheader("Edge selection frequency across 20 seeds")
        st.dataframe(edges, use_container_width=True, hide_index=True)
        shown = edges.sort_values("selection_frequency", ascending=False).head(15)
        shown = shown.assign(edge=shown["source"] + " → " + shown["target"])
        st.bar_chart(shown, x="edge", y="selection_frequency", color="stable", stack=False)
    if role_path.exists():
        st.subheader("Node-role stability")
        st.dataframe(pd.read_csv(role_path), use_container_width=True, hide_index=True)

with robustness_tab:
    robustness_path = CAMERA_TABLE_DIR / "calendar_timing_robustness.csv"
    if robustness_path.exists():
        robustness = pd.read_csv(robustness_path)
        st.subheader("Calendar and timing sensitivity")
        st.dataframe(robustness, use_container_width=True, hide_index=True)
        st.caption("VNI role under alternative data-timing assumptions")
        st.bar_chart(
            robustness, x="variant", y="VNI_net_strength", color="VNI_role", stack=False
        )

with risk_tab:
    risk_path = CAMERA_TABLE_DIR / "gbri_prediction_weights.csv"
    if risk_path.exists():
        risk = pd.read_csv(risk_path, parse_dates=["date"])
        st.subheader("Graph-based Risk Index")
        st.caption(
            "Chỉ segment=test là out-of-sample monitoring; train/validation là "
            "retrospective reconstruction."
        )
        valid = risk.dropna(subset=["gbri"])
        st.line_chart(valid, x="date", y="gbri", color="segment")
        test_risk = valid.loc[valid["segment"].eq("test")]
        st.subheader("Test-period regime distribution")
        st.dataframe(
            test_risk["risk_regime"].value_counts(dropna=False).rename_axis("regime").reset_index(name="days"),
            use_container_width=True,
            hide_index=True,
        )
