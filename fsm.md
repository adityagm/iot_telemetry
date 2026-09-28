# Real-Time IoT Telemetry Finite State Machine (FSM)

## Overview
This architecture implements a streaming Finite State Machine (FSM) to calculate real-time failure probabilities for solar-powered IoT device fleets (~350,000 units). It acts as the Tier 1 low-latency detection layer before passing aggregated features to Tier 2 machine learning models (LightGBM/XGBoost).

---

## Operational States

1. **`HEALTHY`**: Normal operation. Device telemetry parameters are within threshold limits ($P \approx 0.0057\%$).
2. **`WARNING`**: Non-critical transient anomalies detected. Consecutive error streak counter incremented ($P \approx 5\% - 35\%$).
3. **`FAULT_DEEMED`**: Device crossed operational risk thresholds ($P \ge 80\%$) or experienced an acute rate-of-change spike. Triggers an automated remote intervention.
4. **`OFFLINE_SILENT`**: Telemetry missing for $\ge 24$ hours or missing during peak solar daylight hours (10 AM – 4 PM). Triggers a wake/reset remote intervention ($P = 92\%$).
5. **`INTERVENTION_PENDING`**: Device undergoing post-intervention verification. Monitors for $N$ consecutive healthy readings ($P = 40\%$).
6. **`CRITICAL_HARDWARE_FAILURE`**: Remote intervention failed or consecutive faults occurred during the verification window ($P = 99\%$). Automatically dispatches a field repair ticket.

---

## State Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> HEALTHY : Initialized

    HEALTHY --> WARNING : Anomaly Event (Streak < Threshold)
    HEALTHY --> OFFLINE_SILENT : Telemetry Silent (>24h or Solar Daylight Drop)
    HEALTHY --> FAULT_DEEMED : Acute Rate-of-Change Spike (e.g. ΔTemp > 3°C/min)

    WARNING --> HEALTHY : Sensor Normal Range (Streak Reset)
    WARNING --> FAULT_DEEMED : Consecutive Errors >= Threshold OR ML Risk >= 80%
    WARNING --> OFFLINE_SILENT : Telemetry Stops Mid-Anomalous Trend

    OFFLINE_SILENT --> INTERVENTION_PENDING : Trigger Remote Intervention (Wake/Reset Ping)
    OFFLINE_SILENT --> CRITICAL_HARDWARE_FAILURE : Unresponsive > 48h

    FAULT_DEEMED --> INTERVENTION_PENDING : Trigger Remote Intervention

    state INTERVENTION_PENDING {
        [*] --> VERIFYING
        VERIFYING --> VERIFYING : Receiving Healthy Telemetry (Count < N)
    }

    INTERVENTION_PENDING --> HEALTHY : Verification Complete (N Consecutive Healthy Readings)
    INTERVENTION_PENDING --> CRITICAL_HARDWARE_FAILURE : Error Received During Verification

    CRITICAL_HARDWARE_FAILURE --> [*] : Field Technician Repair