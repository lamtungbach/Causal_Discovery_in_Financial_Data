# 📊 NOTEARS Thesis — Causal Discovery & Risk Spillover

## 1. Overview
This project focuses on causal discovery in financial markets using volatility-based time series. The goal is to uncover directional relationships among major assets and quantify risk propagation.

## 2. Problem Statement
- Correlation does not imply causation
- Traditional methods fail with nonlinear dependencies
- Need interpretable causal structures

## 3. Objectives
- Learn DAG structure
- Identify risk transmitters/receivers
- Build Risk Score & Early Warning System

## 4. Data
Assets:
- S&P 500
- VN-Index
- Gold
- WTI Oil
- Bitcoin

Period: 2018–2024

## 5. Methodology
- Neural Granger Causality (cMLP, cLSTM)
- PC Algorithm
- NOTEARS

## 6. Pipeline
1. Data Collection
2. Preprocessing
3. Model Training
4. DAG Extraction
5. Evaluation
6. Risk Scoring
7. Early Warning

## 7. Outputs
- DAG Graph
- Risk Scores
- Spillover Index
- Early Warning Signals

## 8. Contributions
- Apply Neural Granger to finance
- Combine multiple causal methods
- Build causal-based risk system

## 9. Limitations
- No true ground truth
- Sensitive to hyperparameters

## 10. Future Work
- Add more assets
- Use real-time data

