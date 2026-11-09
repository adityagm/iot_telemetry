from pyspark.sql import functions as F
from pyspark.ml.regression import AFTRegression

def train_aft_model(df_model_input, label_col="observed_time", censor_col="event_occurred"):
    """
    Fits Accelerated Failure Time (AFT) survival regression model and outputs RUL predictions.
    """
    aft = AFTRegression(
        featuresCol="features",
        labelCol=label_col,
        censorCol=censor_col,
        predictionCol="predicted_rul_log_scale",
        quantileProbabilities=[0.1, 0.5, 0.9],
        quantilesCol="rul_quantiles"
    )
    
    aft_model = aft.fit(df_model_input)
    df_predictions = aft_model.transform(df_model_input)
    
    # Extract 50th percentile (Median RUL in Days)
    extract_median_rul = F.udf(lambda v: float(v[1]), "double")
    df_final = df_predictions.withColumn("predicted_rul_days", extract_median_rul(F.col("rul_quantiles")))
    
    return aft_model, df_final