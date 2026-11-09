from src.features.feature_engineering import build_rma_features
from src.models.train_aft_survival import train_aft_model
from src.models.evaluate_cindex import compute_c_index_30d_pyspark

# 1. Feature Engineering
df_features = build_rma_features(df_raw)

# 2. Train AFT Regression
model, df_preds = train_aft_model(df_features)

# 3. Evaluate 30-Day C-Index
c_index = compute_c_index_30d_pyspark(df_preds, window_days=30)