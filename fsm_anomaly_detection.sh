# 1. Create the Stream Processor Python file
cat << 'EOF' > stream_processor.py
import json
from datetime import datetime, timezone

class TelemetryStreamProcessor:
    def __init__(self, baseline_prob=0.000286, alert_threshold=0.80, verification_required=3):
        self.baseline_prob = baseline_prob
        self.alert_threshold = alert_threshold
        self.verification_required = verification_required
        self.state_store = {}

    def _get_or_create_device_state(self, device_id, event_time):
        if device_id not in self.state_store:
            self.state_store[device_id] = {
                "last_seen_timestamp": event_time,
                "last_board_temp": None,
                "last_temp_timestamp": None,
                "consecutive_errors": 0,
                "intervention_pending": False,
                "intervention_timestamp": None,
                "healthy_post_intervention_count": 0,
                "intervention_count_historical": 0,
                "current_probability": self.baseline_prob,
                "status": "HEALTHY"
            }
        return self.state_store[device_id]

    def _check_silence_and_solar(self, state, current_time):
        silence_hours = 0.0
        if state["last_seen_timestamp"]:
            delta_seconds = (current_time - state["last_seen_timestamp"]).total_seconds()
            silence_hours = max(0.0, delta_seconds / 3600.0)

        event_hour = current_time.hour
        is_daylight = 10 <= event_hour <= 16
        solar_daylight_drop = is_daylight and (silence_hours >= 4.0)
        return silence_hours, solar_daylight_drop

    def _check_velocity_triggers(self, payload, state, current_time):
        curr_temp = payload.get("internal_board_temperature")
        velocity_spike_detected = False
        temp_delta_per_min = 0.0

        if curr_temp is not None and state["last_board_temp"] is not None and state["last_temp_timestamp"] is not None:
            time_delta_mins = (current_time - state["last_temp_timestamp"]).total_seconds() / 60.0
            if time_delta_mins > 0:
                temp_delta = curr_temp - state["last_board_temp"]
                temp_delta_per_min = temp_delta / time_delta_mins
                if temp_delta_per_min >= 3.0:
                    velocity_spike_detected = True

        if curr_temp is not None:
            state["last_board_temp"] = curr_temp
            state["last_temp_timestamp"] = current_time

        return velocity_spike_detected, temp_delta_per_min

    def process_telemetry_event(self, event_json):
        payload = json.loads(event_json)
        device_id = payload["device_id"]
        current_time = datetime.fromisoformat(payload["timestamp"])
        
        state = self._get_or_create_device_state(device_id, current_time)
        
        silence_hours, solar_daylight_drop = self._check_silence_and_solar(state, current_time)
        velocity_spike, temp_rate_of_change = self._check_velocity_triggers(payload, state, current_time)
        
        errors = []
        if payload.get("battery_voltage", 4.0) < 3.3:
            errors.append("LOW_BATTERY")
        if payload.get("accel_x", 0.0) > 2.0:
            errors.append("HIGH_VIBRATION")
        if payload.get("internal_board_temperature", 30.0) > 45.0:
            errors.append("OVERHEAT")
        if solar_daylight_drop:
            errors.append("SOLAR_DAYLIGHT_SILENCE")
            
        is_error = len(errors) > 0
        action = "NONE"

        if silence_hours >= 24.0 and not state["intervention_pending"]:
            state["status"] = "OFFLINE_SILENT"
            state["current_probability"] = 0.92
            action = "TRIGGER_REMOTE_INTERVENTION (WAKE/RESET PING)"
            state["intervention_pending"] = True
            state["intervention_timestamp"] = current_time.isoformat()
            state["intervention_count_historical"] += 1

        elif velocity_spike:
            state["consecutive_errors"] += 1
            state["status"] = "FAULT_DEEMED (ACUTE_VELOCITY_SPIKE)"
            state["current_probability"] = 0.95
            action = "TRIGGER_REMOTE_INTERVENTION (THERMAL SHUTDOWN)"
            state["intervention_pending"] = True
            state["intervention_timestamp"] = current_time.isoformat()
            state["intervention_count_historical"] += 1

        elif state["intervention_pending"]:
            if not is_error:
                state["healthy_post_intervention_count"] += 1
                if state["healthy_post_intervention_count"] >= self.verification_required:
                    state["intervention_pending"] = False
                    state["consecutive_errors"] = 0
                    state["current_probability"] = 0.0000571
                    state["status"] = "HEALTHY"
                    action = "INTERVENTION_VERIFIED_RESET_TO_HEALTHY"
                else:
                    state["current_probability"] = 0.40
                    action = f"VERIFICATION_IN_PROGRESS ({state['healthy_post_intervention_count']}/{self.verification_required})"
            else:
                state["current_probability"] = 0.99
                state["status"] = "CRITICAL_HARDWARE_FAILURE"
                action = "DISPATCH_FIELD_TECHNICIAN"

        elif is_error:
            state["consecutive_errors"] += 1
            state["healthy_post_intervention_count"] = 0
            
            streak = state["consecutive_errors"]
            if streak == 1:
                state["current_probability"] = 0.05
            elif streak == 2:
                state["current_probability"] = 0.35
            else:
                state["current_probability"] = min(0.80 + (streak - 3) * 0.05, 0.98)

            if state["current_probability"] >= self.alert_threshold:
                state["status"] = "FAULT_DEEMED"
                state["intervention_pending"] = True
                state["intervention_timestamp"] = current_time.isoformat()
                state["intervention_count_historical"] += 1
                action = "TRIGGER_REMOTE_INTERVENTION"
            else:
                state["status"] = "WARNING"
                action = f"LOG_ANOMALY (Streak: {streak})"

        else:
            state["consecutive_errors"] = 0
            state["current_probability"] = 0.0000571
            state["status"] = "HEALTHY"
            action = "NONE"

        state["last_seen_timestamp"] = current_time

        return {
            "device_id": device_id,
            "timestamp": current_time.isoformat(),
            "silence_hours": round(silence_hours, 2),
            "temp_rate_of_change_per_min": round(temp_delta_per_min, 2),
            "event_errors": errors,
            "consecutive_errors": state["consecutive_errors"],
            "failure_probability": round(state["current_probability"], 4),
            "device_status": state["status"],
            "system_action": action
        }
EOF

# 2. Create the FSM Documentation Markdown file
cat << 'EOF' > fsm_theory.md
# Real-Time IoT Telemetry Finite State Machine (FSM)

## Overview
Streaming FSM architecture for calculating real-time failure probabilities across ~350,000 solar-powered IoT devices. Acts as Tier 1 low-latency detection before passing features to Tier 2 ML models (LightGBM/XGBoost).

## Operational States
- `HEALTHY`: Normal operation ($P \approx 0.0057\%$).
- `WARNING`: Non-critical transient anomalies ($P \approx 5\% - 35\%$).
- `FAULT_DEEMED`: Risk threshold breached or acute spike ($P \ge 80\%$). Triggers remote intervention.
- `OFFLINE_SILENT`: Telemetry missing $\ge 24\text{h}$ or solar daylight drop ($P = 92\%$).
- `INTERVENTION_PENDING`: Post-intervention verification monitoring ($P = 40\%$).
- `CRITICAL_HARDWARE_FAILURE`: Verification failed or persistent hard fault ($P = 99\%$). Dispatches field technician.
EOF