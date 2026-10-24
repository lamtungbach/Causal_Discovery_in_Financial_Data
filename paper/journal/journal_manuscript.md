# Identifying Granger Causal Return Spillovers in Multivariate Financial Data with Interpretable Machine Learning

Nguyen Lam Tung Bach, Ngo Minh Huong, and Nguyen Van Tang

Institute for Artificial Intelligence, University of Engineering and Technology, Vietnam National University, Hanoi, Vietnam; Faculty of Technology and Data Science, Foreign Trade University, Hanoi, Vietnam. Corresponding author: tangnv@ftu.edu.vn

## Abstract

This study examines directed predictive return spillovers among the VN-Index and six global financial series while separating forecast attribution from financial-risk measurement. Daily data from January 2015 to May 2026 are processed under a frozen VN-Index calendar and chronological 70/15/15 split. A multivariate VAR(20), an LSTM, and target-specific group-regularized multilayer perceptrons are compared. Neural edges are defined by validation-loss changes after complete source-block ablation; the graph is therefore validation-attribution evidence rather than structural causality. NGC and LSTM forecasts remain statistically indistinguishable. The main q0.70 seed-42 graph contains seven edges, but only VNI -> GOLD reaches the prespecified 0.80 frequency criterion across 20 seeds. An audit shows that 74.12% of test forecast windows contain at least one filled or interpolated input, although only 4.22% of test input-return cells are exposed. Generalized FEVD gives total return connectedness of 22.189% at horizon 10 and 24.405% at horizon 20. The graph-based risk index is positively associated with forward volatility but does not consistently outperform simple baselines. The results support an auditable framework for predictive dependence and exploratory monitoring, not a new architecture, structural causality, or forecast dominance.

**Keywords:** Granger causality; return spillovers; prediction ablation; generalized FEVD; graph stability; VN-Index; risk monitoring.

## 1 Introduction

Financial markets transmit information across asset classes, time zones, and trading calendars. For an emerging market such as Vietnam, the empirical question is not only whether global variables precede domestic returns, but whether the result survives synchronization choices, model class, graph threshold, and random initialization. Correlation measures contemporaneous association. Granger analysis asks whether lagged information improves prediction [1], while connectedness methods allocate forecast-error variance across shocks [5]. These are related but different estimands.

The camera-ready study established a conservative core result: a nonlinear Neural Granger-style predictor did not significantly outperform an LSTM on the held-out VN-Index test set, and the inferred graph was sensitive to seed and calendar design. The journal extension retains that result and adds five analyses without changing raw data, forecast origins, architecture, or camera-ready outputs: cell-level interpolation exposure, fixed threshold sensitivity, finite-seed uncertainty intervals, forward-risk validation of GBRI, and a generalized FEVD benchmark.

Four questions guide the analysis. First, do NGC and LSTM differ in held-out VNI forecast loss or sign accuracy? Second, which validation-attribution edges and node roles persist across thresholds and seeds? Third, what return connectedness is measured by generalized FEVD on the same development sample? Fourth, does the frozen GBRI contain forward-volatility information beyond simple components and persistence baselines?

The contribution is methodological integration rather than a new NGC architecture. The design links data provenance, timing assumptions, paired inference, nonlinear source ablation, threshold and seed uncertainty, generalized FEVD, and prospective index validation. Each empirical conclusion is tied to a saved output and a stated interpretation boundary.

## 2 Related Work

### 2.1 Linear Granger Analysis and Connectedness

Linear Granger causality is commonly implemented through coefficient restrictions in a VAR [1], [2]. Multiple directed tests require multiplicity control; this study applies Benjamini-Hochberg FDR correction [8]. Diebold-Yilmaz connectedness instead summarizes shares of forecast-error variance attributed to cross-variable shocks [5]. We use the generalized Pesaran-Shin decomposition [18] to avoid Cholesky ordering dependence. A VAR-FDR edge, generalized FEVD share, and neural ablation score must not be treated as measurements on one scale.

### 2.2 Neural Granger Causality and Financial Graph Models

Neural Granger Causality represents each target by a nonlinear autoregression and regularizes source-lag blocks [3], [4]. Graph-neural financial models may share parameters across nodes and model dynamic topology [6], [7]. Our target-specific design has a narrower aim: it asks whether removing one standardized source block worsens validation prediction. This gives direct output-based attribution without requiring a prespecified graph, but it does not identify an economic intervention.

### 2.3 Explainable Prediction Attribution

Raw first-layer block norms are not identified under compensating rescaling of adjacent ReLU layers. Complete source-block ablation avoids that scale ambiguity because importance is computed from fitted outputs. The score remains conditional on the model, validation period, zero baseline, and threshold. It is evidence about prediction reliance, not a coefficient, p-value, or structural effect.

### 2.4 Vietnam and Emerging Market Spillovers

Prior research reports return and volatility links between Vietnam and advanced markets using VAR and multivariate GARCH designs [19]. Frequency-domain evidence also finds return transmission from the United States to Vietnam [20]. Those results motivate attention to international dependence, but they do not transfer mechanically to the present multi-asset prediction design. Time-zone alignment and calendar completion matter because one date label can represent different information sets across exchanges.

## 3 Data Provenance and Synchronization Validity

The frozen dataset contains VNI, BTC, OIL, GOLD, SP500, SSE, and VIX. Archived observations run through 11 May 2026. Yahoo Finance adjusted-close series are used for the six global variables; VNI comes from VCI through vnstock. The date is feasible because the pipeline and raw files were produced after that endpoint; no synthetic extension is used. The downloader archive lacks an API retrieval timestamp, so filesystem dates and hashes are retained as provenance evidence.

Returns are percentage log differences defined only when both endpoint prices are finite and positive. The negative WTI close on 20 April 2020 remains in the price audit, while returns ending on 20 and 21 April are invalidated. The main panel is anchored to VNI dates. OIL, GOLD, SP500, and VIX are forward-filled for at most three observations; SSE may also be linearly interpolated for at most five; BTC is not filled.

The complete return panel contains 2,828 observations. Target-date splitting yields 1,959 training, 424 validation, and 425 test forecast origins. Validation and test windows may use prior observations as context, but their targets never enter an earlier split.

### 3.1 Interpolation Exposure

The audit reconstructs the canonical price panel to a maximum absolute difference below 1e-12 and leaves all raw files unchanged. Exposure means that a return endpoint or lag window uses a filled or interpolated price; it is not proof of bias.

| Split | Origins | Exposed cells (%) | Exposed windows (%) |
|---|---:|---:|---:|
| Train | 1,959 | 4.35 | 76.93 |
| Validation | 424 | 5.05 | 76.89 |
| Test | 425 | 4.22 | 74.12 |

In the test split, 315 of 425 windows are exposed. Linear interpolation is confined to SSE. These findings require the main-calendar design to be described as auditable but not interpolation-free. The common-day and extra-timing-shift specifications remain the relevant robustness checks.

## 4 Methods

### 4.1 Chronological Forecast Design

All series are standardized with training-only means and standard deviations. A 20-observation window predicts the next seven-dimensional return vector. The LSTM has 64 hidden units and a linear VNI output. NGC fits one 140-64-64-1 ReLU MLP per target with a group-norm penalty of 5e-4. Adam uses learning rate 1e-3, validation loss selects checkpoints, and test observations are excluded from training, checkpoint selection, graph construction, and GBRI calibration.

Forecast comparisons use the two-sided Harvey-Leybourne-Newbold modified Diebold-Mariano squared-loss test [11], [12], exact McNemar inference for paired sign correctness [13], and the Pesaran-Timmermann directional test [14]. Non-rejection means insufficient evidence of a difference, not equivalence.

### 4.2 VAR FDR and Validation Attribution

A multivariate VAR(20) is fitted on the development period. Each ordered source-target pair receives a joint F test of its 20 lag coefficients, with Benjamini-Hochberg correction across 42 tests at FDR 0.05.

For neural attribution, all 20 standardized lags of one source are replaced by zero on validation inputs. Importance is the nonnegative relative increase in validation MSE. Scores at or above the q0.70 positive-score quantile form the primary graph. q0.60 and q0.80 are fixed sensitivity settings. The quantile controls sparsity and is not a significance threshold. Seed uncertainty uses seeds 42 through 61 with all other settings fixed. Wilson intervals summarize these 20 observed fits, not a market population.

### 4.3 Generalized Forecast Error Variance Decomposition

The FEVD benchmark uses the same development sample and VAR(20) as VAR-FDR. Before row normalization, the generalized share for receiver i and shock j is theta_ij(H) = sigma_jj^(-1) sum_h(e_i' Phi_h Sigma e_j)^2 divided by sum_h(e_i' Phi_h Sigma Phi_h' e_i), for h from 0 to H-1. Rows are receivers and columns are shock sources. FROM is the off-diagonal share received, TO is the contribution sent to other nodes, and NET equals TO minus FROM. Horizons 10 and 20 are prespecified. Because the modeled variables are returns, this is return forecast-error connectedness, not volatility spillover.

### 4.4 Graph Based Risk Index Validation

The frozen GBRI combines absolute external shocks entering VNI in the seed-42 graph with 20-day VNI volatility. Both components use training-reference standardization; alpha is 0.50. It is an exploratory monitoring percentile, not a crash probability or trading signal.

Validation uses only forward 5-day and 10-day realized volatility. A high-volatility event uses the training-only 85th percentile. GBRI is compared with VNI volatility, absolute VIX return, and matching-horizon VNI volatility persistence. Probabilities use a univariate logistic calibrator fitted only on training data. Confidence intervals use 1,000 moving-block bootstrap resamples of fixed test predictions with block length 10; no model is retrained.

## 5 Results

### 5.1 Held Out Forecasting

| Model | RMSE | MAE | R2 | DA (%) |
|---|---:|---:|---:|---:|
| LSTM | 1.237 | 0.824 | 0.024 | 56.00 |
| NGC | 1.246 | 0.823 | 0.010 | 54.59 |

The modified DM statistic is 0.476 with p=0.635. McNemar discordances are 63 LSTM-only correct signs versus 57 NGC-only correct signs, with p=0.648. The evidence does not support a claim that NGC outperforms LSTM.

### 5.2 Threshold Sensitivity

| Quantile | Cutoff | Edges | Incoming VNI | VNI net | VNI role |
|---:|---:|---:|---|---:|---|
| 0.60 | 0.00526 | 10 | SP500; VIX | 0.01065 | Transmitter |
| 0.70 | 0.00547 | 7 | VIX | 0.01596 | Transmitter |
| 0.80 | 0.00669 | 5 | None | 0.02225 | Transmitter |

VNI remains a transmitter at all three settings, but its incoming sources change. q0.70 remains primary because it was specified before this sensitivity analysis, not because it maximizes a VNI result.

![Figure 1. Edge-selection frequency at the three fixed attribution thresholds](../../results/journal_revision/graph_sensitivity/threshold_edge_frequency.png)

### 5.3 Seed Uncertainty

| Edge | Seeds | Frequency | Wilson 95% interval |
|---|---:|---:|---|
| VNI -> GOLD | 18 | 0.90 | [0.699, 0.972] |
| VIX -> VNI | 15 | 0.75 | [0.531, 0.888] |
| OIL -> GOLD | 14 | 0.70 | [0.481, 0.855] |
| SP500 -> VNI | 13 | 0.65 | [0.433, 0.819] |
| BTC -> SSE | 11 | 0.55 | [0.342, 0.742] |

VNI -> GOLD is the only edge meeting the 0.80 consensus criterion. VNI is a transmitter in 13 of 20 seeds and a receiver in 7. Its role is not uniform.

![Figure 2. Wilson intervals for edge-selection frequencies across seeds 42 through 61](../../results/journal_revision/seed_uncertainty/edge_frequency_wilson_intervals.png)

Across 190 unordered seed pairs, mean Jaccard similarity is 0.316 and mean turnover is 0.684. Exact same-seed prediction reproduction and cross-seed graph instability are distinct.

### 5.4 Generalized FEVD Connectedness

| Horizon | TCI (%) | VNI FROM | VNI TO | VNI NET | VNI role |
|---:|---:|---:|---:|---:|---|
| 10 | 22.189 | 19.485 | 7.962 | -11.523 | Receiver |
| 20 | 24.405 | 21.691 | 12.590 | -9.101 | Receiver |

SP500 and VIX are the two largest net transmitters at both horizons. At H=10, SP500 contributes 38.36% and VIX 29.68% of VNI's FROM connectedness. Two variable-order permutations reproduce both matrices after labels are restored, with maximum differences below 4e-16.

![Figure 3. Row-normalized generalized FEVD at horizons 10 and 20](../../results/journal_revision/fevd/gfevd_heatmaps.png)

The FEVD receiver role does not contradict the seed-42 NGC transmitter role. FEVD allocates linear VAR forecast-error variance on the development sample; NGC measures nonlinear prediction reliance on validation data; VAR-FDR tests linear coefficient restrictions. Their magnitudes must not be ranked together.

### 5.5 Forward Risk Validation of GBRI

| Horizon | Model | Spearman | ROC AUC | PR AUC | Brier |
|---:|---|---:|---:|---:|---:|
| 5 | GBRI | 0.361 | 0.657 | 0.265 | 0.118 |
| 5 | VNI volatility | 0.477 | 0.643 | 0.209 | 0.135 |
| 5 | VIX absolute return | 0.020 | 0.595 | 0.228 | 0.116 |
| 5 | Persistence | 0.513 | 0.698 | 0.264 | 0.130 |
| 10 | GBRI | 0.342 | 0.610 | 0.315 | 0.157 |
| 10 | VNI volatility | 0.460 | 0.605 | 0.281 | 0.170 |
| 10 | VIX absolute return | 0.020 | 0.556 | 0.274 | 0.158 |
| 10 | Persistence | 0.530 | 0.698 | 0.361 | 0.156 |

GBRI is positively associated with future realized volatility but does not consistently beat simple baselines. At five days, persistence has higher correlation and ROC AUC. At ten days, persistence again has higher correlation, ROC AUC, and PR AUC, with a marginally lower Brier score. GBRI remains an exploratory monitoring index rather than a validated standalone predictor.

![Figure 4. Forward-risk metrics for GBRI and prespecified baselines](../../results/journal_revision/gbri_validation/gbri_baseline_metrics.png)

## 6 Discussion

### 6.1 Different Estimands Explain Divergent Networks

The three network views do not reveal a hidden winning method. VAR-FDR tests linear restrictions. FEVD allocates fitted forecast-error variance. NGC ablation measures a nonlinear predictor's validation response to source removal. Differences in sample role, null hypothesis, normalization, and perturbation explain divergent edges and node roles.

Some qualitative agreement remains. SP500 and VIX transmit under FEVD and frequently under neural seed analysis. GOLD and SSE are often neural receivers. VNI is the clearest disagreement: it is a seed-42 NGC transmitter, a transmitter in 65% of neural seeds, and a FEVD receiver. This is evidence against describing Vietnam as uniformly one role.

### 6.2 Validity Threats

Calendar completion is material. Most 20-day test windows contain an exposed input, even though exposed cells are a small fraction of all lag cells. The common-day design changes VNI to a weak receiver, while shifting global inputs one additional VNI observation removes retained incoming VNI edges. Daily labels approximate, rather than perfectly identify, information availability across time zones.

q0.70 fixes sparsity but supplies no significance level. Twenty seeds describe optimization variability but do not replace a dependent sampling bootstrap. The validation set selects checkpoints and supplies graph-ablation losses, so the graph is exploratory validation evidence. Zeroing a source block uses the training mean as an attribution baseline.

GBRI validation covers one fixed test period and overlapping forward windows. The moving-block bootstrap reflects local dependence in evaluation but excludes model-retraining uncertainty. The index inherits uncertainty from the frozen graph. Its practical use should remain limited to ranking dates for analyst review.

### 6.3 Practical Interpretation

Each neural edge should be reported with its return target, timing convention, validation-attribution basis, threshold, and seed frequency. FEVD results should state horizon, row normalization, receiver-row orientation, and return-based estimand. GBRI values should be described as training-referenced monitoring percentiles.

The present results do not authorize automated trading, allocation, or regulatory decisions. A future operational study would require rolling-origin estimation, prespecified outcomes, and drift diagnostics. Those changes are outside this frozen extension because they would materially alter the experiment.

## 7 Conclusion

The journal extension preserves the camera-ready data, split, architecture, and core negative findings while adding targeted evidence requested in review. NGC does not significantly outperform LSTM. The neural graph offers validation-attribution evidence but limited edge stability. VNI's role changes across seeds, timing conventions, and estimands. FEVD identifies SP500 and VIX as leading return-connectedness transmitters while VNI is a receiver under that distinct linear decomposition. GBRI is associated with forward volatility but does not consistently beat simple baselines.

The defensible conclusion is bounded: predictive dependencies can be audited when provenance, synchronization exposure, paired inference, attribution sensitivity, and estimand differences are reported together. The evidence does not establish structural causality, uniform receiver status for Vietnam, a new architecture, or forecast dominance.

## Supplementary Appendix Map and Reproducibility

- **Appendix A:** interpolation provenance and exposure under `results/journal_revision/interpolation_audit/`.
- **Appendix B:** q0.60, q0.70, and q0.80 graph sensitivity under `results/journal_revision/graph_sensitivity/`.
- **Appendix C:** seed intervals, consensus graphs, and Jaccard results under `results/journal_revision/seed_uncertainty/`.
- **Appendix D:** forward-risk targets, metrics, and regimes under `results/journal_revision/gbri_validation/`.
- **Appendix E:** generalized FEVD matrices and connectedness tables under `results/journal_revision/fevd/`.
- **Reproducibility:** scripts 17 through 21 generate journal analyses; script 22 performs final consistency checks. The release manifest hashes code, outputs, and manuscript files.

## Acknowledgment

This work was supported by Vietnam National University, Hanoi, under research project QG.26.68.

## References

[1] C. W. J. Granger, "Investigating causal relations by econometric models and cross-spectral methods," Econometrica, 37(3), 424-438, 1969.

[2] H. Lutkepohl, New Introduction to Multiple Time Series Analysis. Springer, 2005.

[3] A. Tank et al., "Neural Granger causality," IEEE TPAMI, 44(8), 4267-4279, 2022.

[4] M. Yuan and Y. Lin, "Model selection and estimation in regression with grouped variables," JRSS B, 68(1), 49-67, 2006.

[5] F. X. Diebold and K. Yilmaz, "Better to give than to receive," International Journal of Forecasting, 28(1), 57-66, 2012.

[6] C. Zhang et al., "Forecasting realized volatility with spillover effects," International Journal of Forecasting, 41(1), 377-397, 2025.

[7] R. Cheng and Q. Li, "Modeling the momentum spillover effect for stock prediction," AAAI, 35(1), 2021.

[8] Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate," JRSS B, 57(1), 289-300, 1995.

[9] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 9(8), 1735-1780, 1997.

[10] D. P. Kingma and J. Ba, "Adam," ICLR, 2015.

[11] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," JBES, 13(3), 253-263, 1995.

[12] D. Harvey et al., "Testing the equality of prediction mean squared errors," International Journal of Forecasting, 13(2), 281-291, 1997.

[13] Q. McNemar, "Note on the sampling error of correlated proportions," Psychometrika, 12(2), 153-157, 1947.

[14] M. H. Pesaran and A. Timmermann, "A simple nonparametric test of predictive performance," JBES, 10(4), 461-465, 1992.

[15] R. S. Tsay, Analysis of Financial Time Series, 2nd ed. Wiley, 2005.

[16] N. Meinshausen and P. Buhlmann, "Stability selection," JRSS B, 72(4), 417-473, 2010.

[17] R. Cont, "Empirical properties of asset returns," Quantitative Finance, 1(2), 223-236, 2001.

[18] M. H. Pesaran and Y. Shin, "Generalised impulse response analysis in linear multivariate models," Economics Letters, 58(1), 17-29, 1998.

[19] X. V. Vo and C. Ellis, "International financial integration," Emerging Markets Review, 36, 19-27, 2018.

[20] M. K. Nguyen and D. N. Le, "Return spillover from the US and Japanese stock markets to Vietnam," Emerging Markets Finance and Trade, 57(1), 47-58, 2021.
