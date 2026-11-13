from pyspark.sql import functions as F
from pyspark.sql.window import Window

def calculate_aft_target_labels(df_daily_features, df_device_events, dataset_cutoff_date="2026-10-31"):
    """
    Calculates labelCol (observed_time) and censorCol (event_occurred) for PySpark AFT Regression.
    
    Parameters:
    - df_daily_features: DataFrame containing columns ['device_id', 'date', ...]
    - df_device_events: DataFrame containing ['device_id', 'rma_date', 'event_type']
    - dataset_cutoff_date: String 'YYYY-MM-DD' representing the max dataset observation date.
    """
    
    # Cast dates to DateType
    df_features = df_daily_features.withColumn("date", F.col("date").cast("date"))
    df_events = df_device_events.withColumn("rma_date", F.col("rma_date").cast("date"))
    
    # 1. Left Join daily features with device event metadata
    df_joined = df_features.join(df_events, on="device_id", how="left")
    
    # 2. Define Dataset Cutoff Literal
    cutoff_lit = F.lit(dataset_cutoff_date).cast("date")
    
    # 3. Calculate event_occurred (censorCol)
    # 1.0 if an RMA occurred AFTER the observation date (t_obs < rma_date <= cutoff)
    # 0.0 if no RMA occurred or if the device remained active up to cutoff
    df_labeled = df_joined.withColumn(
        "event_occurred",
        F.when(
            (F.col("rma_date").isNotNull()) & 
            (F.col("date") < F.col("rma_date")) & 
            (F.col("rma_date") <= cutoff_lit),
            1.0
        ).otherwise(0.0)
    )
    
    # 4. Calculate observed_time (labelCol in days)
    # If event_occurred == 1.0: Days between observation date and RMA date
    # If event_occurred == 0.0: Days between observation date and dataset cutoff date
    df_labeled = df_labeled.withColumn(
        "observed_time",
        F.when(
            F.col("event_occurred") == 1.0,
            F.datediff(F.col("rma_date"), F.col("date")).cast("double")
        ).otherwise(
            F.datediff(cutoff_lit, F.col("date")).cast("double")
        )
    )
    
    # 5. Filter out invalid temporal rows:
    # - Rows on or after the actual rma_date (post-failure telemetry)
    # - Rows where observed_time <= 0 (AFT requires strictly positive time labels)
    df_valid = df_labeled.filter(
        (F.col("observed_time") > 0.0) & 
        (F.col("date") <= cutoff_lit) &
        (F.col("rma_date").isNull() | (F.col("date") < F.col("rma_date")))
    )
    
    return df_valid