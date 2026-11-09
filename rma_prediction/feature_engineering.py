from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler

def build_rma_features(df_raw):
    """
    Computes unified failure risk and builds 90-day window features in PySpark.
    """
    # 1. Unified 24h risk
    df_unified = df_raw.withColumn(
        "p_fault_24h",
        1.0 - ((1.0 - F.col("p_accel")) * (1.0 - F.col("p_temp")) * (1.0 - F.col("p_batt")))
    )

    # 2. Window Definitions
    window_90d = Window.partitionBy("device_id").orderBy("date").rowsBetween(-89, 0)
    window_7d  = Window.partitionBy("device_id").orderBy("date").rowsBetween(-6, 0)
    window_lag = Window.partitionBy("device_id").orderBy("date")

    # 3. Time-Series Aggregations
    df_features = df_unified \
        .withColumn("ewma_30d_p_fault", F.avg("p_fault_24h").over(window_90d)) \
        .withColumn("cum_stress_90d", F.sum("p_fault_24h").over(window_90d)) \
        .withColumn("batt_drain_7d_avg", F.avg("p_batt").over(window_7d)) \
        .withColumn("temp_risk_7d_avg", F.avg("p_temp").over(window_7d)) \
        .withColumn("p_fault_7d_lag", F.lag("p_fault_24h", 7).over(window_lag)) \
        .withColumn("p_fault_velocity", (F.col("p_fault_24h") - F.col("p_fault_7d_lag")) / 7.0) \
        .na.fill(0.0)

    # 4. Vector Assembly
    feature_cols = [
        "p_fault_24h", "ewma_30d_p_fault", "cum_stress_90d",
        "batt_drain_7d_avg", "temp_risk_7d_avg", "p_fault_velocity"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    return assembler.transform(df_features)