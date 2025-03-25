import os
from flask import Flask, request, jsonify, render_template
import pickle
import mlflow
import traceback
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Robust model path resolution
def find_model_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, 'Models', 'best_model.pkl'),
        os.path.join(base_dir, 'models', 'best_model.pkl'),
        os.path.join(base_dir, 'best_model.pkl'),
        '/app/Models/best_model.pkl',
        '/app/models/best_model.pkl',
        '/app/best_model.pkl'
    ]

    for path in possible_paths:
        logger.info(f"Checking model path: {path}")
        if os.path.exists(path):
            logger.info(f"Model found at: {path}")
            return path

    logger.error("No model file found in any expected location")
    return None


# Load model and stats
model = None
scaler = None
try:
    model_path = find_model_file()

    if model_path:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)

        model = model_dict['model']
        scaler = model_dict['scaler']
    else:
        logger.error("Could not locate model file")

except Exception as e:
    logger.error(f"Model Loading Error: {e}")
    logger.error(traceback.format_exc())


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        logger.error("Model not loaded correctly")
        return jsonify({
            "error": "Model not loaded correctly. Check server logs.",
            "model_status": "Not Loaded",
            "working_directory": os.getcwd(),
            "files_in_directory": os.listdir('.')
        }), 500

    try:
        # Existing prediction logic remains the same
        data = request.get_json()

        if not data:
            logger.error("No input data received")
            return jsonify({
                "error": "No input data received"
            }), 400

        feature_order = [
            'credit_lines_outstanding',
            'loan_amt_outstanding',
            'total_debt_outstanding',
            'income',
            'years_employed',
            'fico_score'
        ]

        # Validate input data
        for feature in feature_order:
            if feature not in data:
                logger.error(f"Missing feature: {feature}")
                return jsonify({
                    "error": f"Missing required feature: {feature}"
                }), 400

        # Convert features to float and handle potential conversion errors
        try:
            features = [float(data.get(feature, 0)) for feature in feature_order]
        except ValueError as ve:
            logger.error(f"Feature conversion error: {ve}")
            return jsonify({
                "error": "Invalid feature values. All features must be numeric."
            }), 400

        # Perform prediction
        features_scaled = scaler.transform([features])
        prob = model.predict_proba(features_scaled)[0][1]
        prediction = 1 if prob > 0.3 else 0

        return jsonify({
            "prediction": int(prediction),
            "probability": float(prob),
            "message": "High risk of default!" if prediction == 1 else "Low risk. Loan can be granted."
        })

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Unexpected error during prediction",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)