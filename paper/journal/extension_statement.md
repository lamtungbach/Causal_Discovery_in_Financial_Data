# Conference to Journal Extension Statement

The journal manuscript retains the frozen seven-series dataset, target-date split, VAR(20), LSTM and NGC-style MLP specifications, prediction-based seed-42 graph, paired forecast inference, common-day and timing robustness checks, and training-referenced GBRI construction. It does not present a new neural architecture and does not revise camera-ready artifacts.

The journal extension adds five bounded analyses: a cell-level interpolation exposure audit; fixed q0.60, q0.70, and q0.80 threshold sensitivity; Wilson intervals, consensus graphs, and pairwise Jaccard summaries for the existing 20 seeds; forward-risk validation of the frozen GBRI at prespecified 5-day and 10-day horizons; and row-normalized generalized FEVD at horizons 10 and 20 on the same VAR development sample.

The manuscript also expands related work, estimand definitions, validity threats, and reproducibility documentation. Negative findings remain visible: NGC does not significantly outperform LSTM, only VNI -> GOLD reaches the 0.80 neural consensus criterion, VNI is not uniformly a receiver, and GBRI does not consistently beat simple baselines.
