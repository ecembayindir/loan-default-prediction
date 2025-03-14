# eda_preprocessing.py
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os

# File paths
data_path = "Data/Loan_Data.csv"
output_path = "Data/Processed_Loan_Data.csv"
stats_path = "Data/Loan_Data_Describe.csv"

# Load data
df = pd.read_csv(data_path)
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())

# Basic statistics
print("\nDescriptive Statistics:\n", df.describe())
df.describe().to_csv(stats_path)

# Visualizations (save as PNG instead of displaying)
# Default Distribution
fig = px.histogram(df, x="default", title="Default Distribution", color="default")
fig.write_image("Data/default_distribution.png")
print("Default distribution plot saved to Data/default_distribution.png")

# Box plots
for col in ["income", "fico_score", "years_employed", "loan_amt_outstanding"]:
    fig = px.box(df, x=col, color="default", title=f"{col} by Default")
    fig.write_image(f"Data/{col}_by_default.png")
    print(f"{col} by default plot saved to Data/{col}_by_default.png")

# Correlation Matrix
corr_matrix = df.corr().round(2)
fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Blues', title="Correlation Matrix")
fig.write_image("Data/correlation_matrix.png")
print("Correlation matrix plot saved to Data/correlation_matrix.png")

# Feature importance (using matplotlib, which is more reliable for static saving)
from sklearn.ensemble import RandomForestClassifier
X = df.drop(columns=["default", "customer_id"])
y = df["default"]
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feat_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feat_importance)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance")
plt.savefig("Data/feature_importance.png")
plt.close()  # Close the plot to free memory
print("Feature importance plot saved to Data/feature_importance.png")

# Preprocessing
df.drop(columns=["customer_id"], inplace=True)
df["default"] = df["default"].astype(bool)
num_cols = ["loan_amt_outstanding", "total_debt_outstanding", "income", "fico_score", "years_employed"]
df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
X = df.drop(columns=["default"])
y = df["default"]
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name="default")], axis=1)
df_balanced.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")

# Git commit
os.system('git add . && git commit -m "EDA and preprocessing completed with static plots"')
os.system('git push origin main')