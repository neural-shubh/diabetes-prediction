from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

FEATURES = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    values = []
    for f in FEATURES:
        val = data.get(f, 0)
        if isinstance(val, bool):
            values.append(1 if val else 0)
        else:
            values.append(float(val))

    x = np.array(values).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({
        'result': prediction,
        'confidence': confidence,
        'positive_prob': round(float(proba[list(model.classes_).index('Positive')]) * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)
