from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask('__name__')

# Load the model (replace with your actual file path)
with open('fraud_detection_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Create input data for prediction (as a nested list)
        input_data = [[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]

        # Preprocess input data
        input_data = scaler.transform(input_data)
        input_data = imputer.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]

        result = "Fraudulent" if prediction == 0 else "Fraudulent"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':  # Corrected conditional statement
    app.run(debug=True)