# Result Traceability

| Claim | Manuscript value | Generated source |
|---|---|---|
| C01 | 2,828 returns; 1,959/424/425 forecast origins | `results/camera_ready/tables/main_split_summary.csv` |
| C02 | LSTM RMSE 1.237; NGC RMSE 1.246 | `results/camera_ready/tables/forecast_metrics.csv` |
| C03 | Modified DM 0.476, p=0.635 | `results/camera_ready/tables/forecast_inference.json` |
| C04 | McNemar 63 versus 57, p=0.648 | `results/camera_ready/tables/forecast_inference.json` |
| C05 | Test exposure 4.22% of cells and 74.12% of windows | `results/journal_revision/interpolation_audit/window_exposure_summary.csv` |
| C06 | Seed-42 q0.60/q0.70/q0.80 graphs have 10/7/5 edges | `results/journal_revision/graph_sensitivity/threshold_graph_summary.csv` |
| C07 | VNI transmitter 13/20 and receiver 7/20 | `results/journal_revision/seed_uncertainty/node_role_frequency_intervals.csv` |
| C08 | VNI -> GOLD 18/20, frequency 0.90, Wilson [0.699, 0.972] | `results/journal_revision/seed_uncertainty/edge_frequency_intervals.csv` |
| C09 | GBRI Spearman 0.361 at 5 days and 0.342 at 10 days | `results/journal_revision/gbri_validation/gbri_baseline_metrics.csv` |
| C10 | TCI 22.189% at H=10 and 24.405% at H=20 | `results/journal_revision/fevd/connectedness_summary.csv` |
| C11 | VNI FEVD NET -11.523 at H=10 and -9.101 at H=20 | `results/journal_revision/fevd/connectedness_summary.csv` |
| C12 | H=10 VNI incoming shares: SP500 38.36%, VIX 29.68% | `results/journal_revision/fevd/vni_connectedness.csv` |

Figures are linked directly to generated PNG files under `results/journal_revision/`. Tables are transcribed from the listed machine-readable outputs and verified by `scripts/22_verify_journal_outputs.py`.
