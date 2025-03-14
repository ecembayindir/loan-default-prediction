# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import mlflow
import os

# Set MLflow tracking URI (same as training)
mlflow.set_tracking_uri("http://localhost:8080")  # Update to remote URI in production

# Start an MLflow experiment for production monitoring
mlflow.set_experiment("Production_Monitoring")

# Load model and stats
model_path = "Models/best_model.pkl"
stats_path = "Data/Loan_Data_Describe.csv"
model = pickle.load(open(model_path, "rb"))
stats = pd.read_csv(stats_path).loc[[1, 2]].reset_index(drop=True)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Start an MLflow run for each prediction
    with mlflow.start_run(run_name="prediction_run"):
        data = request.get_json()
        credit_lines = float(data['credit_lines_outstanding'])
        loan_amt = float(data['loan_amt_outstanding'])
        total_debt = float(data['total_debt_outstanding'])
        income = float(data['income'])
        years_employed = float(data['years_employed'])
        fico_score = float(data['fico_score'])

        # Normalize features
        features = [
            credit_lines,
            (loan_amt - stats.iloc[0, 1]) / stats.iloc[1, 1],
            (total_debt - stats.iloc[0, 2]) / stats.iloc[1, 2],
            (income - stats.iloc[0, 3]) / stats.iloc[1, 3],
            (years_employed - stats.iloc[0, 4]) / stats.iloc[1, 4],
            (fico_score - stats.iloc[0, 5]) / stats.iloc[1, 5],
        ]

        # Predict probability
        prob = model.predict_proba([features])[0][1]
        prediction = 1 if prob > 0.5 else 0

        # Log inputs and outputs to MLflow
        mlflow.log_param("credit_lines_outstanding", credit_lines)
        mlflow.log_param("loan_amt_outstanding", loan_amt)
        mlflow.log_param("total_debt_outstanding", total_debt)
        mlflow.log_param("income", income)
        mlflow.log_param("years_employed", years_employed)
        mlflow.log_param("fico_score", fico_score)
        mlflow.log_metric("prediction_probability", prob)
        mlflow.log_metric("prediction_label", prediction)

        # Return response
        return jsonify({
            "prediction": prediction,
            "probability": prob,
            "message": "High risk of default!" if prob > 0.5 else "Low risk. Loan can be granted."
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)