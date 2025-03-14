# model_training.py
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# File paths
data_path = "Data/Processed_Loan_Data.csv"
model_path = "Models/best_model.pkl"
random_seed = 42
mlflow.set_tracking_uri("http://localhost:8080")

# Load data
df = pd.read_csv(data_path)
X = df.drop("default", axis=1)
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Logging function
def log_model(model, X_test, y_test, y_pred, params, run_name, experiment_name):
    with mlflow.start_run(run_name=run_name, experiment_id=mlflow.set_experiment(experiment_name).experiment_id):
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = {
            "precision": class_report["1"]["precision"],
            "recall": class_report["1"]["recall"],
            "f1_score": class_report["1"]["f1-score"],
            "false_positives": int(conf_matrix[0, 1]),
            "false_negatives": int(conf_matrix[1, 0]),
        }
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(conf_matrix, display_labels=["No Default", "Default"]).plot(ax=ax)
        plt.title(f"Confusion Matrix - {run_name}")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(fig, "confusion_matrix.png")
        mlflow.sklearn.log_model(model, run_name)
    return metrics

# Logistic Regression
mlflow.set_experiment("Logistic_Regression")
params_lr = {"class_weight": "balanced", "solver": "saga", "penalty": "elasticnet", "l1_ratio": 0.5, "max_iter": 1000,
             "random_state": random_seed}
lr = LogisticRegression(**params_lr)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics_lr = log_model(lr, X_test, y_test, y_pred_lr, params_lr, "lr_balanced_saga", "Logistic_Regression")
print("Logistic Regression Metrics:", metrics_lr)

# Random Forest
mlflow.set_experiment("Random_Forest")
params_rf = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": random_seed}
rf = RandomForestClassifier(**params_rf)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
metrics_rf = log_model(rf, X_test, y_test, y_pred_rf, params_rf, "rf_depth10_est100", "Random_Forest")
print("Random Forest Metrics:", metrics_rf)

# Save best model
best_model = rf if metrics_rf["f1_score"] > metrics_lr["f1_score"] else lr
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {model_path}")

# Git commit
os.system('git add . && git commit -m "Model training and MLflow tracking completed"')
os.system('git push origin main')