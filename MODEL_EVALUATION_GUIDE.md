# BTC-Predictor: Model Performance & Evaluation Framework

This document outlines the key performance indicators (KPIs) used to evaluate the Bitcoin forecasting model. It distinguishes between **Training/Validation** (learning capability) and **Backtesting** (trading profitability).

---

## 1. Training & Validation Metrics
*Focus: Does the model understand the mathematical structure of the market?*

| Metric | Measurement | "Great" Range | Impact of Poor Score |
| :--- | :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | The average distance between prediction and reality. | **Decreasing** or stable across folds. | High MAE indicates the model is failing to find any predictive signal (blindness). |
| **Interval Coverage (80%)** | How often the actual price stays within the P10-P90 "cone." | **75% - 85%** | < 60%: Overconfident (dangerous). <br> > 90%: Too cautious (useless). |
| **Directional Accuracy (DA)** | Percentage of correct Up/Down predictions. | **> 53%** | ~50% is a coin flip. In crypto, 55% is considered "institutional grade." |
| **Pinball Loss** | Specific error metric for quantile (probabilistic) forecasts. | **Lower** than baseline models. | High loss means the model doesn't understand the probability of extreme moves. |

---

## 2. Backtesting & Simulation Metrics
*Focus: Does the model's logic translate into profit after fees and slippage?*

| Metric | Measurement | "Great" Range | The Reality Check |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | Risk-adjusted return (Return / Volatility). | **> 1.5** | 1.0 is the minimum for viability. <br> **1.63 (Current LightGBM Score)** is excellent. |
| **Max Drawdown (MDD)** | The largest peak-to-valley percentage loss. | **< 15%** | If MDD is high, you risk liquidation or stopping the bot during a period of "pain." |
| **Turnover** | Frequency of portfolio rebalancing/trading. | **Low to Moderate** | High turnover results in "Death by Fees," where all profits go to the exchange. |
| **Profit Factor** | Ratio of Gross Profit to Gross Loss. | **> 1.5** | Tells you if your wins are large enough to cover your inevitable losses. |
| **Expectancy** | Expected profit per trade ($). | **Positive** | Must be high enough to cover the "Trading Threshold" (Fees + Slippage). |

---

## 3. The "Overfitting" Test
To ensure the model is robust, we compare the two sections:

*   **Scenario A (Healthy):** High Directional Accuracy in Training **leads to** a high Sharpe Ratio in Backtesting. (Current State)
*   **Scenario B (Overfit):** High Directional Accuracy in Training **leads to** a negative Sharpe Ratio in Backtesting. (Model memorized history but cannot generalize).

---

## 4. Current Model Status (brutally honest)
- **Engineering:** 9/10 (Leakage-safe, Chained trajectories).
- **Data Alpha:** 7/10 (High impact from Open Interest/Funding).
- **Profitability:** 1h and 6h horizons show strong Alpha; 4h and 12h are currently mean-reverting.

---
*Generated on: 2026-01-01*
