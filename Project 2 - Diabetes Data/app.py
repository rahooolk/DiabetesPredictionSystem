# app.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request
from diabetes_model import DiabetesModel

app = Flask(__name__, static_folder='static', template_folder='templates')
diabetes_model = DiabetesModel()  # Initialize your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])

    # Use your machine learning model for prediction
    scaled_features = diabetes_model.scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    print("Input Values:", [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    print("Scaled Features:", scaled_features)

    print("Feature Coefficients:", diabetes_model.model.coef_)

    #prediction = diabetes_model.predict(scaled_features)
    prediction = (diabetes_model.model.predict_proba(scaled_features)[:, 1] > 0.3).astype(int)
    print("Prediction:", prediction)

    result = "Positive" if prediction == 1 else "Negative"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
