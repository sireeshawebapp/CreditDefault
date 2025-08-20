from flask import Flask, render_template, request
import joblib
import numpy as np
import urllib.request
import io

app = Flask(__name__)

# The public URL for your model file in Google Cloud Storage
MODEL_URL = "https://storage.googleapis.com/creditdefaulters/final_credit_default_model_rf.pkl"

# Load the model directly from the URL
try:
    with urllib.request.urlopen(MODEL_URL) as response:
        model_bytes = response.read()
        model = joblib.load(io.BytesIO(model_bytes))
except Exception as e:
    print(f"Error loading model from URL: {e}")
    model = None # Set model to None to prevent app from crashing

@app.route("/")
def index():
    if model is None:
        return "Model not loaded. Please check logs.", 500
    return render_template("index.html", user_input={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form

        features = [
            float(user_input["LIMIT_BAL"]),
            int(user_input["SEX"]),
            int(user_input["EDUCATION"]),
            int(user_input["MARRIAGE"]),
            int(user_input["AGE"]),
            float(user_input["TOTAL_BILL_AMT"]),
            float(user_input["TOTAL_PAY_AMT"]),
            float(user_input["AVG_BILL_AMT"]),
            float(user_input["AVG_PAY_AMT"]),
            float(user_input["UTILIZATION_RATIO"])
        ]

        # Use the loaded model to make predictions
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0][1]

        return render_template("index.html",
                               prediction=prediction,
                               probability=round(prob, 3),
                               user_input=user_input)

    except Exception as e:
        return render_template("index.html", error=str(e), user_input=request.form)

if __name__ == "__main__":
    app.run(debug=True)