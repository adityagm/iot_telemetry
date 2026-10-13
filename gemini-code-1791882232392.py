from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Max allowable time discrepancy (e.g., 6 hours in seconds)
MAX_TOLERANCE_SECONDS = 6 * 3600 

# -------------------------------------------------------------------------
# Step 1: Clean & Align Schemas Before Union
# -------------------------------------------------------------------------

# Filter out unlinked assets to prevent severe Spark partition skew
df_telemetry_clean = (
    df_telemetry
    .filter(F.col("asset_id").isNotNull() & F.col("event_time").isNotNull())
    .select(
        F.col("asset_id"),
        F.col("device_id"),
        F.col("event_time").cast("timestamp").alias("event_time"),
        F.lit(None).cast("string").alias("event_name"),
        F.col("batt_voltage").cast("double").alias("batt_voltage"),
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
        F.lit(1).alias("is_telemetry")  # Flag to distinguish row source
    )
)

df_booking_clean = (
    df_booking
    .filter(F.col("asset_id").isNotNull() & F.col("event_time").isNotNull())
    .select(
        F.col("asset_id"),
        F.lit(None).cast("string").alias("device_id"),
        F.col("event_time").cast("timestamp").alias("event_time"),
        F.col("event_name").cast("string").alias("event_name"),
        F.lit(None).cast("double").alias("batt_voltage"),
        F.lit(None).cast("double").alias("latitude"),
        F.lit(None).cast("double").alias("longitude"),
        F.lit(0).alias("is_telemetry")
    )
)

# -------------------------------------------------------------------------
# Step 2: Union and Create Forward & Backward Windows
# -------------------------------------------------------------------------
df_combined = df_booking_clean.unionByName(df_telemetry_clean)

# Forward window (looking at previous telemetry)
w_forward = (
    Window.partitionBy("asset_id")
    .orderBy("event_time")
    .rowsBetween(Window.unboundedPreceding, 0)
)

# Backward window (looking at next telemetry)
w_backward = (
    Window.partitionBy("asset_id")
    .orderBy("event_time")
    .rowsBetween(0, Window.unboundedFollowing)
)

# Extract previous and next telemetry metadata
df_bracketed = (
    df_combined
    # Previous Telemetry (Forward Fill candidate)
    .withColumn("prev_time", F.last(F.when(F.col("is_telemetry") == 1, F.col("event_time")), ignorenulls=True).over(w_forward))
    .withColumn("prev_volt", F.last(F.when(F.col("is_telemetry") == 1, F.col("batt_voltage")), ignorenulls=True).over(w_forward))
    .withColumn("prev_lat", F.last(F.when(F.col("is_telemetry") == 1, F.col("latitude")), ignorenulls=True).over(w_forward))
    .withColumn("prev_long", F.last(F.when(F.col("is_telemetry") == 1, F.col("longitude")), ignorenulls=True).over(w_forward))
    .withColumn("prev_dev", F.last(F.when(F.col("is_telemetry") == 1, F.col("device_id")), ignorenulls=True).over(w_forward))
    
    # Next Telemetry (Backward Fill candidate)
    .withColumn("next_time", F.first(F.when(F.col("is_telemetry") == 1, F.col("event_time")), ignorenulls=True).over(w_backward))
    .withColumn("next_volt", F.first(F.when(F.col("is_telemetry") == 1, F.col("batt_voltage")), ignorenulls=True).over(w_backward))
    .withColumn("next_lat", F.first(F.when(F.col("is_telemetry") == 1, F.col("latitude")), ignorenulls=True).over(w_backward))
    .withColumn("next_long", F.first(F.when(F.col("is_telemetry") == 1, F.col("longitude")), ignorenulls=True).over(w_backward))
    .withColumn("next_dev", F.first(F.when(F.col("is_telemetry") == 1, F.col("device_id")), ignorenulls=True).over(w_backward))
)

# -------------------------------------------------------------------------
# Step 3: Filter Down to Booking Events and Apply Bounded Selection
# -------------------------------------------------------------------------

df_milestones = (
    df_bracketed
    .filter(F.col("is_telemetry") == 0)  # Keep only milestone events
    .withColumn("diff_prev_sec", F.abs(F.col("event_time").cast("long") - F.col("prev_time").cast("long")))
    .withColumn("diff_next_sec", F.abs(F.col("next_time").cast("long") - F.col("event_time").cast("long")))
    # Determine the closest valid telemetry ping
    .withColumn(
        "closest_direction",
        F.when(
            F.col("diff_prev_sec").isNull() & F.col("diff_next_sec").isNull(), F.lit("NONE")
        ).when(
            F.col("diff_prev_sec").isNull(), F.lit("NEXT")
        ).when(
            F.col("diff_next_sec").isNull(), F.lit("PREV")
        ).when(
            F.col("diff_prev_sec") <= F.col("diff_next_sec"), F.lit("PREV")
        ).otherwise(F.lit("NEXT"))
    )
    .withColumn(
        "min_time_diff_sec",
        F.when(F.col("closest_direction") == "PREV", F.col("diff_prev_sec"))
         .when(F.col("closest_direction") == "NEXT", F.col("diff_next_sec"))
         .otherwise(None)
    )
    # Check if telemetry falls within acceptable operational tolerance
    .withColumn("is_within_tolerance", F.col("min_time_diff_sec") <= MAX_TOLERANCE_SECONDS)
    
    # Assign attributes based on nearest validated ping
    .withColumn(
        "resolved_device_id",
        F.when(~F.col("is_within_tolerance"), None)
         .when(F.col("closest_direction") == "PREV", F.col("prev_dev"))
         .otherwise(F.col("next_dev"))
    )
    .withColumn(
        "resolved_batt_voltage",
        F.when(~F.col("is_within_tolerance"), None)
         .when(F.col("closest_direction") == "PREV", F.col("prev_volt"))
         .otherwise(F.col("next_volt"))
    )
    .withColumn(
        "resolved_latitude",
        F.when(~F.col("is_within_tolerance"), None)
         .when(F.col("closest_direction") == "PREV", F.col("prev_lat"))
         .otherwise(F.col("next_lat"))
    )
    .withColumn(
        "resolved_longitude",
        F.when(~F.col("is_within_tolerance"), None)
         .when(F.col("closest_direction") == "PREV", F.col("prev_long"))
         .otherwise(F.col("next_long"))
    )
    .select(
        "asset_id",
        "resolved_device_id",
        "event_name",
        "event_time",
        "resolved_batt_voltage",
        "resolved_latitude",
        "resolved_longitude",
        F.round(F.col("min_time_diff_sec") / 3600.0, 2).alias("telemetry_offset_hours"),
        "is_within_tolerance"
    )
)