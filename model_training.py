import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
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

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_seed)


def log_model(model, X_test, y_test, y_pred, params, run_name, experiment_name, scaler=None):
    with mlflow.start_run(run_name=run_name, experiment_id=mlflow.set_experiment(experiment_name).experiment_id):
        # More robust class report extraction
        pos_label = list(set(y_test.unique()) - {min(y_test.unique())})[0]
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')

        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = {
            "precision": class_report_dict[str(pos_label)]["precision"],
            "recall": class_report_dict[str(pos_label)]["recall"],
            "f1_score": class_report_dict[str(pos_label)]["f1-score"],
            "false_positives": int(conf_matrix[0, 1]),
            "false_negatives": int(conf_matrix[1, 0]),
            "cross_val_f1_mean": np.mean(cv_scores),
            "cross_val_f1_std": np.std(cv_scores)
        }

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(conf_matrix, display_labels=sorted(y_test.unique())).plot(ax=ax)
        plt.title(f"Confusion Matrix - {run_name}")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(fig, "confusion_matrix.png")

        # Optional: log model with scaler
        if scaler:
            mlflow.sklearn.log_model(
                {'model': model, 'scaler': scaler},
                run_name,
                signature=mlflow.models.signature.infer_signature(X_test)
            )
        else:
            mlflow.sklearn.log_model(model, run_name)

    return metrics


# Logistic Regression
mlflow.set_experiment("Logistic_Regression")
params_lr = {
    "class_weight": "balanced",
    "solver": "saga",
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "max_iter": 2000,
    "random_state": random_seed
}
lr = LogisticRegression(**params_lr)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics_lr = log_model(lr, X_test, y_test, y_pred_lr, params_lr, "lr_balanced_saga", "Logistic_Regression", scaler)
print("Logistic Regression Metrics:", metrics_lr)

# Random Forest
mlflow.set_experiment("Random_Forest")
params_rf = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": random_seed,
    "class_weight": "balanced_subsample"
}
rf = RandomForestClassifier(**params_rf)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
metrics_rf = log_model(rf, X_test, y_test, y_pred_rf, params_rf, "rf_depth10_est100", "Random_Forest")
print("Random Forest Metrics:", metrics_rf)

# Select best model
best_model_metrics = max([metrics_lr, metrics_rf], key=lambda x: x['f1_score'])
best_model = lr if best_model_metrics == metrics_lr else rf

# Save best model and scaler
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({'model': best_model, 'scaler': scaler}, f)
print(f"Best model saved to {model_path}")

# Optional: Safer git commit with error handling
try:
    os.system('git add .')
    commit_result = os.system('git commit -m "Model training and MLflow tracking completed"')
    if commit_result == 0:
        os.system('git push origin main')
    else:
        print("No changes to commit.")
except Exception as e:
    print(f"Git operations failed: {e}")