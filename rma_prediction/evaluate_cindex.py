from pyspark.sql import functions as F

def compute_c_index_30d_pyspark(
    df_preds, 
    label_col="observed_time", 
    censor_col="event_occurred", 
    pred_col="predicted_rul_days", 
    window_days=30,
    num_partitions=200
):
    """
    Computes Harrell's Concordance Index in PySpark bounded to a 30-day event window.
    """
    df_clean = df_preds.select(
        F.col(label_col).cast("double").alias("T"),
        F.col(censor_col).cast("double").alias("E"),
        F.col(pred_col).cast("double").alias("pred")
    ).repartition(num_partitions)

    df_i = df_clean.filter(F.col("E") == 1.0) \
        .withColumnRenamed("T", "T_i").withColumnRenamed("pred", "pred_i").select("T_i", "pred_i")

    df_j = df_clean \
        .withColumnRenamed("T", "T_j").withColumnRenamed("pred", "pred_j").select("T_j", "pred_j")

    join_condition = (F.col("T_i") < F.col("T_j")) & (F.col("T_j") <= F.col("T_i") + window_days)
    pairs_df = df_i.join(df_j, join_condition)

    scored_pairs = pairs_df.withColumn(
        "score",
        F.when(F.col("pred_i") < F.col("pred_j"), 1.0)
         .when(F.col("pred_i") == F.col("pred_j"), 0.5)
         .otherwise(0.0)
    )

    metrics = scored_pairs.agg(
        F.sum("score").alias("concordant_sum"),
        F.count("score").alias("total_pairs")
    ).collect()[0]

    c_index = metrics["concordant_sum"] / metrics["total_pairs"]
    return c_index