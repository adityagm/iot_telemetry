# Comprehensive IoT Device Failure Prediction & Operational Framework

This document aggregates the complete methodology, statistical formulations, machine learning pipelines, real-time streaming state engine, and operational strategies for modeling device failure probabilities across a fleet of **350,000 solar-powered IoT devices**.

---

## Table of Contents
1. [Problem Formulation & Fleet Parameters](#1-problem-formulation--fleet-parameters)
2. [Statistical Baseline & Conditional Probabilities](#2-statistical-baseline--conditional-probabilities)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [State Machine Logic & Feature Store Engineering](#4-state-machine-logic--feature-store-engineering)
5. [Machine Learning Pipeline Implementation (Batch/LightGBM)](#5-machine-learning-pipeline-implementation-batchlightgbm)
6. [Real-Time Telemetry Streaming & FSM Implementation (Python)](#6-real-time-telemetry-streaming--fsm-implementation-python)
7. [Hybrid System Integration (Tier-1 FSM + Tier-2 ML)](#7-hybrid-system-integration-tier-1-fsm--tier-2-ml)

---

## 1. Problem Formulation & Fleet Parameters

### Scale & Data Context
* **Active Fleet Size ($N_{\text{devices}}$):** 350,000 deployed solar-powered IoT devices.
* **Sensor Payload:** 8 distinct sensors (Accelerometer $X/Y/Z$, Battery Voltage, Solar Panel Voltage, Internal Board Temperature, Ambient Temperature, Door Open Sensor, Speed, Light Sensor).
* **Ingestion Volume:** ~1,000,000 events/24 hours across the fleet.
* **Fault Telemetry:** ~1,000 fault events daily originating from ~100 distinct failing devices.
* **Study Period:** 30 days of continuous telemetry observation.
* **Target Label ($Y_{i, t+1}$):** $Y_{i, t+1} \in \{0, 1\}$. Set to $1$ if device $i$ experiences a critical fault condition requiring intervention on day $t+1$, otherwise $0$.
* **Imbalance Class Ratio:** $\approx 0.0285\%$ daily positive rate per device ($\sim 100 / 350,000$).

### Core Domain Rules
1. **Consecutive Error Rule:** A single isolated error event is treated as noise. A failure condition requires $N$ consecutive error readings across a short time window.
2. **State Reset Condition:** Clearing a fault requires an explicit **remote reboot command**, followed by a **sustained sequence of verified healthy sensor readings** within expected operational ranges.

---

## 2. Statistical Baseline & Conditional Probabilities

### Step 1: Fleet Baseline Probability ($P_{\text{base}}$)
Calculating the generic probability of any random device failing tomorrow across the entire study period:

$$\text{Total Fleet-Days} = \text{Devices} \times \text{Study Days} = 350,000 \times 30 = 10,500,000 \text{ device-days}$$

$$\text{Total Failures} = 100 \text{ failing devices/day} \times 30 \text{ days} = 3,000 \text{ total failure instances}$$

$$P(\text{Failure Tomorrow}_{\text{Baseline}}) = \frac{3,000}{10,500,000} \approx 0.0002857 \quad (\mathbf{0.0286\%})$$

### Step 2: Conditional Probability (Behavioral Segmentation)
Segmenting the fleet into two operational groups based on day $t$ behavior:

* **Group A (Warning Group - ~100 devices/day logging consecutive faults):**
  Assuming ~80% of devices entering this consecutive fault state completely fail or require intervention on day $t+1$:
  $$P(\text{Fail Tomorrow} \mid \text{Logged Faults Today}) = \frac{80}{100} = \mathbf{80.0\%}$$

* **Group B (Healthy Group - ~349,900 devices/day with zero or isolated noise events):**
  $$P(\text{Fail Tomorrow} \mid \text{No Faults Today}) = \frac{20}{349,900} \approx \mathbf{0.0057\%}$$

| Device Status Today | Daily Count | Calculation | Next-Day Failure Risk |
| :--- | :--- | :--- | :--- |
| **Random Device (Baseline)** | 350,000 | $\frac{3,000}{10,500,000}$ | **0.0286%** (~1 in 3,500) |
| **Healthy Device (Group B)** | 349,900 | $\frac{20}{349,900}$ | **0.0057%** (~1 in 17,500) |
| **Warning Device (Group A)** | 100 | $\frac{80}{100}$ | **80.0%** (8 in 10) |

---

## 3. End-to-End System Architecture

```
 Raw Telemetry Stream (1M Daily Events)
                 │
                 ├──► [ Tier 1: Real-Time Streaming FSM Engine ]
                 │     ├── Updates In-Memory Counters / Streak Flags
                 │     ├── Evaluates Immediate Safety Breaches
                 │     └── Action: Instant Automated Remote Reboot
                 │
                 └──► [ Tier 2: Daily Feature Store Aggregator ]
                       ├── Generates Rolling 1d, 3d, 7d Aggregates
                       ├── Fits Calibrated LightGBM Classifier
                       └── Action: Generates Daily Top-N Dispatch Queue & SHAP Root-Cause
```

---

## 4. State Machine Logic & Feature Store Engineering

### Finite State Machine (FSM) Lifecycle
Every device transitions through four states managed by real-time stream state tracking:

$$\text{Healthy} \xrightarrow{\text{Consecutive Errors}} \text{Degrading/Faulty} \xrightarrow{\text{Remote Reboot Sent}} \text{Reboot Verification} \xrightarrow{\text{3x Healthy Readings}} \text{Healthy}$$

### Aggregated Feature Store Definitions
Features generated at midnight (00:00 UTC) for batch model training and inference:

* **Solar & Power Health:**
  * `solar_v_max`: Maximum daily solar panel voltage recorded.
  * `batt_v_min`: Minimum daily battery voltage recorded.
  * `solar_anomaly_count`: Count of times solar voltage was low ($<2.0\text{V}$) during peak sunlight ($>400\text{ lux}$).
* **Thermal Stress:**
  * `temp_diff_max`: $\max(\text{internal\_board\_temp} - \text{ambient\_temp})$. Detects thermal runaway during charging.
* **Error Streaks & Persistence:**
  * `consec_batt_faults_roll_3d`: 3-day rolling sum of consecutive low-voltage faults.
  * `consec_vibe_faults_roll_3d`: 3-day rolling sum of accelerometer anomalies.
  * `reboot_count_roll_7d`: Cumulative count of remote reboots executed in the last 7 days.

---

## 5. Machine Learning Pipeline Implementation (Batch/LightGBM)

This script generates synthetic telemetry, builds rolling feature windows, trains a class-weighted LightGBM model, applies Isotonic Probability Calibration, and outputs an operational top-$N$ inspection queue.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, auc
import datetime

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. GENERATE SYNTHETIC TELEMETRY DATA
# ==========================================
print("Generating synthetic IoT events...")
n_devices = 500  # Scaled demonstration sample
n_days = 30
start_date = datetime.date(2026, 8, 1)

data = []
for day in range(n_days):
    curr_date = start_date + datetime.timedelta(days=day)
    for dev_id in range(1000, 1000 + n_devices):
        n_events = np.random.randint(2, 7)
        for _ in range(n_events):
            solar_v = np.random.uniform(0.0, 18.0)
            batt_v = np.random.uniform(3.2, 4.2)
            amb_temp = np.random.uniform(15.0, 42.0)
            
            # Inject conditional fault behaviors for specific devices
            is_failing_device = (dev_id % 25 == 0) and (day > 15)
            if is_failing_device:
                batt_v = np.random.uniform(2.8, 3.3)
                accel_x = np.random.uniform(2.5, 8.0) # High vibration
                door_open = np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                accel_x = np.random.uniform(0.0, 0.5)
                door_open = np.random.choice([0, 1], p=[0.9, 0.1])

            data.append({
                "device_id": f"DEV_{dev_id}",
                "event_local_time": pd.Timestamp(curr_date) + pd.Timedelta(hours=np.random.randint(0, 24)),
                "date": curr_date,
                "accel_x": accel_x,
                "accel_y": np.random.uniform(0.0, 0.5),
                "accel_z": np.random.uniform(9.6, 10.0),
                "battery_voltage": batt_v,
                "solar_panel_voltage": solar_v,
                "internal_board_temperature": amb_temp + np.random.uniform(2.0, 8.0),
                "door_open_sensor": door_open,
                "container_inside_temperature": np.random.uniform(-4.0, 4.0),
                "ambient_temperature": amb_temp,
                "speed_sensor": np.random.uniform(0.0, 80.0),
                "light_sensor": np.random.uniform(100, 800),
                "is_reboot_event": 1 if (is_failing_device and np.random.rand() > 0.7) else 0
            })

df_raw = pd.DataFrame(data)
df_raw = df_raw.sort_values(by=["device_id", "event_local_time"]).reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING (Daily Aggregations)
# ==========================================
print("Building daily feature matrix...")

df_raw["flag_batt_low"] = (df_raw["battery_voltage"] < 3.3).astype(int)
df_raw["flag_high_vibe"] = (df_raw["accel_x"] > 2.0).astype(int)
df_raw["flag_overheat"] = (df_raw["internal_board_temperature"] > 45.0).astype(int)
df_raw["flag_solar_drop"] = ((df_raw["solar_panel_voltage"] < 2.0) & (df_raw["light_sensor"] > 400)).astype(int)

daily_df = df_raw.groupby(["device_id", "date"]).agg(
    daily_events=("event_local_time", "count"),
    batt_v_min=("battery_voltage", "min"),
    batt_v_mean=("battery_voltage", "mean"),
    solar_v_max=("solar_panel_voltage", "max"),
    temp_diff_max=("internal_board_temperature", lambda x: (x - df_raw.loc[x.index, "ambient_temperature"]).max()),
    vibe_max=("accel_x", "max"),
    consec_batt_faults=("flag_batt_low", "sum"),
    consec_vibe_faults=("flag_high_vibe", "sum"),
    consec_overheat_faults=("flag_overheat", "sum"),
    solar_anomaly_count=("flag_solar_drop", "sum"),
    reboot_count=("is_reboot_event", "sum")
).reset_index()

daily_df = daily_df.sort_values(by=["device_id", "date"]).reset_index(drop=True)

# Rolling Historical Features
for col in ["consec_batt_faults", "consec_vibe_faults", "reboot_count", "solar_anomaly_count"]:
    daily_df[f"{col}_roll_3d"] = daily_df.groupby("device_id")[col].transform(lambda x: x.rolling(3, min_periods=1).sum())
    daily_df[f"{col}_roll_7d"] = daily_df.groupby("device_id")[col].transform(lambda x: x.rolling(7, min_periods=1).sum())

# Define Target: Critical failure condition on Day T+1
daily_df["total_daily_faults"] = (
    daily_df["consec_batt_faults"] + daily_df["consec_vibe_faults"] + daily_df["consec_overheat_faults"]
)
daily_df["target_next_day_failure"] = (
    daily_df.groupby("device_id")["total_daily_faults"]
    .shift(-1)
    .apply(lambda x: 1 if x >= 3 else 0)
)

daily_df = daily_df.dropna(subset=["target_next_day_failure"]).copy()

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================
dates = sorted(daily_df["date"].unique())
train_dates = dates[:20]
val_dates = dates[20:25]
test_dates = dates[25:]

train_df = daily_df[daily_df["date"].isin(train_dates)]
val_df = daily_df[daily_df["date"].isin(val_dates)]
test_df = daily_df[daily_df["date"].isin(test_dates)]

feature_cols = [c for c in daily_df.columns if c not in ["device_id", "date", "target_next_day_failure", "total_daily_faults"]]

X_train, y_train = train_df[feature_cols], train_df["target_next_day_failure"]
X_val, y_val = val_df[feature_cols], val_df["target_next_day_failure"]
X_test, y_test = test_df[feature_cols], test_df["target_next_day_failure"]

# ==========================================
# 4. MODEL TRAINING & CALIBRATION
# ==========================================
scale_pos = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    scale_pos_weight=scale_pos,
    random_state=42,
    verbosity=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_raw_preds = model.predict_proba(X_val)[:, 1]
test_raw_preds = model.predict_proba(X_test)[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_raw_preds, y_val)

test_calibrated_preds = calibrator.transform(test_raw_preds)

precision, recall, _ = precision_recall_curve(y_test, test_calibrated_preds)
pr_auc = auc(recall, precision)

print(f"\nModel PR-AUC Score: {pr_auc:.4f}")

# Operational Priority Output
latest_day = max(test_dates)
latest_ops_df = test_df[test_df["date"] == latest_day].copy()
latest_ops_df["failure_probability"] = calibrator.transform(model.predict_proba(latest_ops_df[feature_cols])[:, 1])

top_queue = latest_ops_df.sort_values(by="failure_probability", ascending=False)[
    ["device_id", "failure_probability", "batt_v_min", "reboot_count_roll_7d", "consec_batt_faults_roll_3d"]
].head(10)

print(f"\n--- Operational Queue: Top High-Risk Devices for Day {latest_day} ---")
print(top_queue.to_string(index=False))
```

---

## 6. Real-Time Telemetry Streaming & FSM Implementation (Python)

This script processes streaming JSON telemetry in real time, manages in-memory device states, dynamically updates probabilities as events arrive, and triggers automated system actions.

```python
import time
import json
import random
from datetime import datetime, timezone

# ==========================================
# 1. REAL-TIME STREAM PROCESSOR & FSM
# ==========================================
class TelemetryStreamProcessor:
    def __init__(self, baseline_prob=0.000286, alert_threshold=0.80):
        self.baseline_prob = baseline_prob
        self.alert_threshold = alert_threshold
        self.state_store = {}

    def _get_or_create_device_state(self, device_id):
        if device_id not in self.state_store:
            self.state_store[device_id] = {
                "consecutive_errors": 0,
                "reboot_pending": False,
                "reboot_timestamp": None,
                "healthy_post_reboot_count": 0,
                "current_probability": self.baseline_prob,
                "status": "HEALTHY"
            }
        return self.state_store[device_id]

    def _evaluate_event_health(self, payload):
        errors = []
        if payload.get("battery_voltage", 4.0) < 3.3:
            errors.append("LOW_BATTERY")
        if payload.get("accel_x", 0.0) > 2.0:
            errors.append("HIGH_VIBRATION")
        if payload.get("internal_board_temperature", 30.0) > 45.0:
            errors.append("OVERHEAT")
        return len(errors) > 0, errors

    def process_telemetry_event(self, event_json):
        payload = json.loads(event_json)
        device_id = payload["device_id"]
        timestamp = payload["timestamp"]
        
        state = self._get_or_create_device_state(device_id)
        is_error, error_types = self._evaluate_event_health(payload)

        # CASE A: Device awaiting post-reboot verification
        if state["reboot_pending"]:
            if not is_error:
                state["healthy_post_reboot_count"] += 1
                if state["healthy_post_reboot_count"] >= 3:
                    state["reboot_pending"] = False
                    state["consecutive_errors"] = 0
                    state["current_probability"] = 0.0000571
                    state["status"] = "HEALTHY"
                    action = "RECOVERY_VERIFIED_RESET_TO_HEALTHY"
                else:
                    state["current_probability"] = 0.40
                    action = f"REBOOT_VERIFICATION_IN_PROGRESS ({state['healthy_post_reboot_count']}/3)"
            else:
                state["current_probability"] = 0.99
                state["status"] = "CRITICAL_HARDWARE_FAILURE"
                action = "DISPATCH_FIELD_TECHNICIAN"

        # CASE B: Anomaly Detected
        elif is_error:
            state["consecutive_errors"] += 1
            state["healthy_post_reboot_count"] = 0
            
            streak = state["consecutive_errors"]
            if streak == 1:
                state["current_probability"] = 0.05
            elif streak == 2:
                state["current_probability"] = 0.35
            else:
                state["current_probability"] = min(0.80 + (streak - 3) * 0.05, 0.98)

            if state["current_probability"] >= self.alert_threshold and not state["reboot_pending"]:
                state["status"] = "FAULT_DEEMED"
                state["reboot_pending"] = True
                state["reboot_timestamp"] = timestamp
                action = "ISSUE_REMOTE_REBOOT_COMMAND"
            else:
                state["status"] = "WARNING"
                action = f"LOG_ANOMALY (Streak: {streak})"

        # CASE C: Normal Reading
        else:
            state["consecutive_errors"] = 0
            state["current_probability"] = 0.0000571
            state["status"] = "HEALTHY"
            action = "NONE"

        return {
            "device_id": device_id,
            "timestamp": timestamp,
            "event_errors": error_types,
            "consecutive_errors": state["consecutive_errors"],
            "failure_probability": round(state["current_probability"], 4),
            "device_status": state["status"],
            "system_action": action
        }

# ==========================================
# 2. EXECUTE REAL-TIME SIMULATION
# ==========================================
if __name__ == "__main__":
    processor = TelemetryStreamProcessor(alert_threshold=0.80)
    target_device = "DEV_88921"

    print("--- Real-Time Stream Processing Execution ---\n")

    stream_sequence = [
        ("normal", {"battery_voltage": 3.8, "accel_x": 0.1, "internal_board_temperature": 32.0}),
        ("fault",  {"battery_voltage": 3.1, "accel_x": 2.5, "internal_board_temperature": 48.0}), # Error 1
        ("fault",  {"battery_voltage": 3.0, "accel_x": 2.8, "internal_board_temperature": 49.0}), # Error 2
        ("fault",  {"battery_voltage": 2.9, "accel_x": 3.1, "internal_board_temperature": 51.0}), # Error 3 -> Threshold Met!
        ("normal", {"battery_voltage": 3.8, "accel_x": 0.1, "internal_board_temperature": 33.0}), # Reboot Recovery 1
        ("normal", {"battery_voltage": 3.8, "accel_x": 0.1, "internal_board_temperature": 32.5}), # Reboot Recovery 2
        ("normal", {"battery_voltage": 3.8, "accel_x": 0.1, "internal_board_temperature": 32.0})  # Reboot Recovery 3 -> Reset
    ]

    for event_type, sensors in stream_sequence:
        now = datetime.now(timezone.utc).isoformat()
        payload = {"device_id": target_device, "timestamp": now, **sensors}
        
        result = processor.process_telemetry_event(json.dumps(payload))
        
        print(f"Event Received: {sensors}")
        print(f"  └── Risk Score: {result['failure_probability']*100:.2f}% | "
              f"Status: {result['device_status']} | "
              f"Action: {result['system_action']}")
        print("-" * 75)
        time.sleep(0.2)
```

---

## 7. Hybrid System Integration (Tier-1 FSM + Tier-2 ML)

The real-time streaming FSM and batch ML model function together in a unified architecture:

1. **State Store Feed:** The streaming FSM maintains live state counters in memory ($O(1)$ lookup). At midnight, these states are snapshotted into the data warehouse as features (`consec_batt_faults_roll_3d`, `reboot_count_roll_7d`).
2. **Model Scoring Calibration:** The batch ML model computes multi-day sensor interaction risk. Its calibrated outputs overwrite simple static FSM rules, enabling the stream engine to evaluate complex risk scores dynamically.
3. **Operational Triage Pipeline:**
   * **Tier 1 Action (Immediate):** Fast-path automated remote reboots triggered by real-time FSM streak limits ($< 10\text{ ms}$).
   * **Tier 2 Action (Next-Day):** Dispatch queues generated by the ML model for physical field inspections, prioritized by SHAP feature attribution.