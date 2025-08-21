from flask import Flask, render_template, request
import joblib
import numpy as np
import urllib.request
import io

app = Flask(__name__)

# Public URL for your trained model file
MODEL_URL = "https://storage.googleapis.com/creditdefaulters/final_credit_default_model_rf.pkl"

# Load the model directly from the URL
try:
    with urllib.request.urlopen(MODEL_URL) as response:
        model_bytes = response.read()
        model = joblib.load(io.BytesIO(model_bytes))
except Exception as e:
    print(f"Error loading model from URL: {e}")
    model = None


@app.route("/")
def index():
    if model is None:
        return "Model not loaded. Please check logs.", 500
    return render_template("index.html", user_input={})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form

        # Required inputs from form
        limit_bal = float(user_input["LIMIT_BAL"])
        sex = int(user_input["SEX"])
        education = int(user_input["EDUCATION"])
        marriage = int(user_input["MARRIAGE"])
        age = int(user_input["AGE"])
        total_bill_amt = float(user_input["TOTAL_BILL_AMT"])
        total_pay_amt = float(user_input["TOTAL_PAY_AMT"])

        # Derived features (calculated automatically)
        avg_bill_amt = total_bill_amt / 6 if total_bill_amt > 0 else 0
        avg_pay_amt = total_pay_amt / 6 if total_pay_amt > 0 else 0
        utilization_ratio = total_bill_amt / limit_bal if limit_bal > 0 else 0

        # Feature vector for prediction
        features = [
            limit_bal, sex, education, marriage, age,
            total_bill_amt, total_pay_amt,
            avg_bill_amt, avg_pay_amt, utilization_ratio
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0][1]

        return render_template(
            "index.html",
            prediction="Default" if prediction == 1 else "No Default",
            probability=round(prob, 3),
            user_input=user_input
        )

    except Exception as e:
        return render_template("index.html", error=str(e), user_input=request.form)


if __name__ == "__main__":
    app.run(debug=True)
