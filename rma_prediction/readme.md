# Maritime IoT Predictive Maintenance & RMA Risk Framework

An enterprise end-to-end Machine Learning and Survival Analysis pipeline built in PySpark and Plotly to predict Remaining Useful Life (RUL) and identify Return Merchandise Authorization (RMA) candidates across large-scale IoT fleets (e.g., maritime telematics tracking devices).

## Executive Overview
In global logistics and shipping asset tracking, device failure typically manifests as either sensor degradation or complete operational blackouts (non-reporting/lost-damaged-removed units). This framework converts raw daily 24-hour probabilities across accelerometer, temperature, and battery drain sensors into:
1. **Unified 24-Hour Risk Index ($P_{\text{fault}}$):** Non-linear combined risk score.
2. **Accelerated Failure Time (AFT) RUL Regression:** PySpark ML survival modeling to handle right-censored fleet data.
3. **Bounded 30-Day Concordance Evaluation:** Scalable PySpark Harrell's C-Index evaluation.
4. **ISO 13374 Risk Zone Visualizations:** Enterprise Plotly charts for dashboard integration.

---

## Mathematical Architecture

### 1. Joint Daily Probability ($P_{\text{fault}}$)
Assuming independent component failure modes across sensors $A$ (Accelerometer), $T$ (Temperature), and $B$ (Battery):

$$P_{\text{fault}, 24\text{h}} = 1 - (1 - P_A)(1 - P_T)(1 - P_B)$$

### 2. Accelerated Failure Time (AFT) Survival Model
To prevent right-censoring bias (active units at sea), failure time $Y$ is modeled against feature matrix $\mathbf{X}$:

$$\log(Y) = \boldsymbol{\beta}^T \mathbf{X} + \sigma W$$

### 3. Harrell's 30-Day Bounded Concordance Index
Evaluates pairwise ranking accuracy strictly within a 30-day operational horizon ($T_j - T_i \le 30$):

$$C_{\text{30d}} = \frac{\sum_{i, j} \mathbb{I}(\widehat{\text{RUL}}_i < \widehat{\text{RUL}}_j)}{\text{Total Comparable Pairs}}$$

---

## Quick Start & Usage

```bash
# Clone and install dependencies
git clone [https://github.com/your-org/your-repo.git](https://github.com/your-org/your-repo.git)
cd your-repo
pip install -r requirements.txt