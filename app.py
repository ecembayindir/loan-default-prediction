# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import mlflow
import os
from contextlib import nullcontext

# Set MLflow tracking URI (disable in production if no server is available)
try:
    mlflow.set_tracking_uri("http://localhost:8080")  # Update to remote URI in production
except (ConnectionError, mlflow.exceptions.MlflowException) as e:
    print(f"MLflow tracking URI not available: {e}. Running without MLflow logging.")

# Start an MLflow experiment for production monitoring (optional)
try:
    mlflow.set_experiment("Production_Monitoring")
except (ConnectionError, mlflow.exceptions.MlflowException) as e:
    pass

# Load model and stats
try:
    model_path = "Models/best_model.pkl"
    stats_path = "Data/Loan_Data_Describe.csv"
    model = pickle.load(open(model_path, "rb"))
    stats = pd.read_csv(stats_path).loc[[1, 2]].reset_index(drop=True)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required files not found: {e}") from e
except (pickle.PickleError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
    raise RuntimeError(f"Error loading model or stats: {e}") from e

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Start an MLflow run for each prediction (if available)
        with mlflow.start_run(run_name="prediction_run") if 'mlflow' in globals() else nullcontext():
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

            # Log inputs and outputs to MLflow (if available)
            if 'mlflow' in globals():
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
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except (AttributeError, IndexError) as e:
        # More specific exceptions for data processing errors
        return jsonify({"error": f"Data processing error: {e}"}), 500
    except RuntimeError as e:
        # More specific than general Exception
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    import socket
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", port))
            sock.close()
            break
        except OSError:
            print(f"Port {port} is in use, trying {port + 1}")
            port += 1
    app.run(host="0.0.0.0", port=port, debug=True)
    print(f"Running on port {port}")