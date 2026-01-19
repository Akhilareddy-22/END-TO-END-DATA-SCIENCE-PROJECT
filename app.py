from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from frontend

# Load trained model
model = pickle.load(open("model/trained_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        area = float(data["area"])
        bedrooms = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])

        prediction = model.predict([[area, bedrooms, bathrooms]])
        price = round(prediction[0], 2)

        return jsonify({
            "success": True,
            "prediction": price,
            "message": f"Predicted House Price: ₹ {price}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# Serve the frontend
@app.route('/frontend')
def serve_frontend():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
