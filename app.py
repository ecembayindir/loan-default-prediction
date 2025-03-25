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

    # Correctly load the model and scaler from the dictionary
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    model = model_dict['model']  # Extract the actual model
    scaler = model_dict['scaler']  # Extract the scaler

    # Debug print to verify model and scaler
    print("Model Type:", type(model))
    print("Scaler Type:", type(scaler))
    print("Model Attributes:", dir(model))
    print("Scaler Attributes:", dir(scaler))

except Exception as e:
    print(f"Model Loading Error: {e}")
    raise

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

            feature_order = [
                'credit_lines_outstanding',
                'loan_amt_outstanding',
                'total_debt_outstanding',
                'income',
                'years_employed',
                'fico_score'
            ]

            features = [float(data.get(feature, 0)) for feature in feature_order]

            features_scaled = scaler.transform([features])

            # Get detailed probability information
            prob = model.predict_proba(features_scaled)[0][1]

            # Lower threshold for prediction (adjust as needed)
            prediction = 1 if prob > 0.3 else 0

            return jsonify({
                "prediction": int(prediction),
                "probability": float(prob),
                "message": "High risk of default!" if prediction == 1 else "Low risk. Loan can be granted."
            })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

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
