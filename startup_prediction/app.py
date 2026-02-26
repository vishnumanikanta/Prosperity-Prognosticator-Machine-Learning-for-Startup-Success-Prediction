from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["age_first_funding_year"]),
        float(request.form["age_last_funding_year"]),
        float(request.form["age_first_milestone_year"]),
        float(request.form["age_last_milestone_year"]),
        float(request.form["relationships"]),
        float(request.form["funding_rounds"]),
        float(request.form["funding_total_usd"]),
        float(request.form["milestones"]),
        float(request.form["avg_participants"])
    ]

    data = np.array([values])
    data = scaler.transform(data)

    proba = model.predict_proba(data)[0]
    print("Probabilities:", proba)

    pred = model.predict(data)[0]

    if pred == "acquired":
     result = f"Acquired ({proba.max()*100:.1f}%)"
    else:
     result = f"Closed ({proba.max()*100:.1f}%)"



    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
