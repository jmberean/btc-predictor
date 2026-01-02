# BTC-Predictor: The Path to "Perfect World" Performance

This document outlines the findings from a deep-dive code review and provides a strategic roadmap to evolve the current "Professional Grade" (Sharpe ~1.6) system into a "Perfect World" (Sharpe > 3.0) algorithmic trading engine.

---

## 1. Deep Dive Findings & Bottlenecks

### A. The "Frequency Mismatch" (Structural Blockade)
*   **Current State:** The system operates on **1-hour bars**.
*   **The Flaw:** High-Alpha signals (Liquidation Cascades, OI Squeezes) are ephemeral events that resolve in 10-20 minutes. By the time a 1-hour bar closes, the market has already repriced the information.
*   **Impact:** The model is permanently "late" to the strongest moves, forcing it to capture only the tail end of trends.

### B. The "Target Rigidity" (Forecasting vs. Trading)
*   **Current State:** The model predicts the **Close Price** `log_return`.
*   **The Flaw:** Traders do not survive on Close prices; they survive on Path Dependency (High/Low). A prediction of +1% is useless if the price wicked -5% (hitting stops) before rallying.
*   **Impact:** The Backtest metrics (Sharpe 1.63) likely overestimate real-world performance because they assume we don't get stopped out by intra-bar volatility.

### C. The "Missing Dimension" (Intent)
*   **Current State:** Features include Price, Volume, and Open Interest.
*   **The Flaw:** We see *Activity* (Volume) but not *Aggressiveness* (Delta). We don't know if the volume was aggressive market buying (bullish) or limit filling (neutral).
*   **Impact:** The model struggles to differentiate between a "Fakeout" (High Volume, Low Delta) and a "Breakout" (High Volume, High Delta).

---

## 2. "Perfect World" Roadmap

To achieve Sharpe > 3.0, Max Drawdown < 5%, and Win Rate > 60%, we must implement the following phases:

### Phase 1: The High-Frequency Upgrade (Immediate ROI)
**Goal:** Capture alpha before it decays.
*   **Action:** Transition the entire pipeline from **1h** to **15m** resolution.
*   **Implementation:**
    1.  Update `configs/binance_bulk.yaml` to `timeframe: 15m`.
    2.  Scale feature windows in `engineering.py` (e.g., `rolling(24)` becomes `rolling(96)`).
    3.  Resync data at 15m granularity.
*   **Expected Metric Impact:** Sharpe 1.6 $ightarrow$ 2.0+. Directional Accuracy 53% $ightarrow$ 56%.

### Phase 2: The "Manager" (Meta-Labeling)
**Goal:** Filter out noise to boost Win Rate.
*   **Action:** Build a secondary model that predicts *when to trade*.
*   **Implementation:**
    1.  Train a Random Forest classifier.
    2.  **Input:** Market Volatility + Primary Model Confidence.
    3.  **Output:** Probability of Trade Success.
    4.  **Rule:** Only take trades where Meta-Model Probability > 65%.
*   **Expected Metric Impact:** Max Drawdown -0.7% $ightarrow$ -0.2%. Win Rate 53% $ightarrow$ 60%.

### Phase 3: Triple-Barrier Modeling
**Goal:** Align model math with trader reality.
*   **Action:** Switch from Regression (predicting value) to Classification (predicting outcome).
*   **Implementation:**
    1.  Define Targets: **[Hit Profit, Hit Stop, Timeout]**.
    2.  Refactor `train.py` to use `LGBMClassifier`.
    3.  Trade only when "Hit Profit" probability is dominant.
*   **Expected Metric Impact:** Profit Factor 1.2 $ightarrow$ 2.0+.

---

## 3. Recommended Immediate Next Step

**Execute Phase 1 (15m Shift).**
It requires the least architectural refactoring while providing the highest probability of immediate metric improvement.

*Generated on: 2026-01-01*
