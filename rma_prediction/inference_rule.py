Before running calculate_aft_target_labels, inspect each RMA'd device's sensor probabilities in the 14-day window prior to rma_date:Sensor- or Battery-Driven RMA (event_occurred = 1.0): Any RMA device where at least one sensor probability (p_batt, p_temp, or p_accel) exceeded a risk threshold (e.g., $\max(P_{\text{fault}, 24\text{h}}) \ge 0.50$ or battery drain spike) within the 14 days preceding rma_date. These represent predictable internal degradation.Sudden Physical LSD (event_occurred = 0.0, Right-Censored at rma_date): Any RMA device whose telemetry remained completely healthy ($\max(P_{\text{fault}, 24\text{h}}) < 0.50$) right up until the day it abruptly stopped reporting or was ticketed as lost/damaged.PySpark Implementation (No event_type Required)Replace target_labeling.py with this self-contained function that derives event_occurred and observed_time using only ['device_id', 'date', 'p_accel', 'p_temp', 'p_batt'] and ['device_id', 'rma_date']:

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def calculate_aft_target_labels_inferred(
    df_daily_telemetry: DataFrame,
    df_device_events: DataFrame,
    dataset_cutoff_date: str = "2026-10-31",
    pre_rma_lookback_days: int = 14,
    fault_signal_threshold: float = 0.50,
) -> DataFrame:
    """
    Infers predictable hardware/battery failures vs. sudden physical LSD directly from
    pre-RMA telemetry without requiring an explicit 'event_type' column.

    Parameters:
    - df_daily_telemetry: ['device_id', 'date', 'p_accel', 'p_temp', 'p_batt', ...]
    - df_device_events: ['device_id', 'rma_date'] (rma_date is null for active units)
    - dataset_cutoff_date: Max observation date in dataset ('YYYY-MM-DD')
    - pre_rma_lookback_days: Window before rma_date to check for sensor/battery degradation
    - fault_signal_threshold: Peak unified risk threshold confirming internal hardware failure
    """
    df_tel = df_daily_telemetry.withColumn("date", F.col("date").cast("date")).withColumn(
        "p_fault_24h",
        1.0 - ((1.0 - F.col("p_accel")) * (1.0 - F.col("p_temp")) * (1.0 - F.col("p_batt"))),
    )
    df_ev = df_device_events.withColumn("rma_date", F.col("rma_date").cast("date"))
    cutoff_lit = F.lit(dataset_cutoff_date).cast("date")

    # 1. Identify which RMA devices showed pre-failure sensor/battery signals in the lookback window
    df_pre_rma_check = (
        df_tel.join(df_ev.filter(F.col("rma_date").isNotNull()), on="device_id", how="inner")
        .filter(
            (F.col("date") >= F.date_sub(F.col("rma_date"), pre_rma_lookback_days))
            & (F.col("date") < F.col("rma_date"))
        )
        .groupBy("device_id")
        .agg(F.max("p_fault_24h").alias("peak_pre_rma_risk"))
        .withColumn(
            "is_predictable_failure",
            F.when(F.col("peak_pre_rma_risk") >= fault_signal_threshold, 1.0).otherwise(0.0),
        )
        .select("device_id", "is_predictable_failure")
    )

    # 2. Join telemetry with RMA dates and inferred failure flag
    df_joined = df_tel.join(df_ev, on="device_id", how="left").join(
        df_pre_rma_check, on="device_id", how="left"
    )

    # 3. Assign censorCol ('event_occurred')
    # 1.0 only if RMA occurred before cutoff AND device exhibited pre-RMA fault degradation
    df_labeled = df_joined.withColumn(
        "event_occurred",
        F.when(
            (F.col("rma_date").isNotNull())
            & (F.col("is_predictable_failure") == 1.0)
            & (F.col("date") < F.col("rma_date"))
            & (F.col("rma_date") <= cutoff_lit),
            1.0,
        ).otherwise(0.0),
    )

    # 4. Assign labelCol ('observed_time' in days)
    # Devices with sudden physical LSD exit at rma_date with event_occurred = 0.0 (right-censored)
    effective_end_date = F.when(
        (F.col("rma_date").isNotNull()) & (F.col("rma_date") <= cutoff_lit),
        F.col("rma_date"),
    ).otherwise(cutoff_lit)

    df_labeled = df_labeled.withColumn(
        "observed_time",
        F.datediff(effective_end_date, F.col("date")).cast("double"),
    )

    # 5. Sanitize rows: exclude post-RMA dates and enforce strictly positive durations (> 0)
    df_valid = df_labeled.filter(
        (F.col("observed_time") > 0.0)
        & (F.col("date") <= cutoff_lit)
        & (F.col("rma_date").isNull() | (F.col("date") < F.col("rma_date")))
    ).drop("is_predictable_failure")

    return df_valid



How to Calibrate fault_signal_threshold on Your 1,500 RMAs

Run a quick distribution check on peak_pre_rma_risk across your 1,500 historical RMA devices before training:

    You will typically see a bimodal split: roughly 40%–55% of RMAs will show elevated pre-RMA risk (peak_pre_rma_risk >= 0.45–0.60), while the remaining sudden LSD / physical loss units sit near baseline (< 0.25).

    Set fault_signal_threshold at the valley between those two peaks (around 0.45–0.50) so your AFT model learns cleanly from true internal degradation without being corrupted by unpredicted physical losses.