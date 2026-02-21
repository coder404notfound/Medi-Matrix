from flask import Flask, render_template, request, jsonify
import joblib
import os

import numpy as np

app = Flask(__name__)

# Define the directories containing your model files
DIABETES_MODEL_DIR = 'models/diabetes_models/'
HEART_DISEASE_MODEL_DIR = 'models/heart_disease_models/'

# Load the models dynamically into separate dictionaries
def load_models():
    models = {
        'diabetes': {},
        'heart_disease': {}
    }

    # Load diabetes models
    for filename in os.listdir(DIABETES_MODEL_DIR):
        if filename.endswith('.pkl'):
            filepath = os.path.join(DIABETES_MODEL_DIR, filename)
            models['diabetes'][filename] = joblib.load(filepath)

    # Load heart disease models
    for filename in os.listdir(HEART_DISEASE_MODEL_DIR):
        if filename.endswith('.pkl'):
            filepath = os.path.join(HEART_DISEASE_MODEL_DIR, filename)
            models['heart_disease'][filename] = joblib.load(filepath)

    return models

# Initialize models at startup
models = load_models()

@app.route('/')
def home():
    return render_template('index.html', models=models)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    return render_template('diabetes.html', models=models)

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    return render_template('heart_disease.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    selected_model = data.get('model')

    if selected_model in models['diabetes']:
        features = [
            float(data.get('Pregnancies')),
            float(data.get('Glucose')),
            float(data.get('BloodPressure')),
            float(data.get('SkinThickness')),
            float(data.get('Insulin')),
            float(data.get('BMI')),
            float(data.get('DiabetesPedigreeFunction')),
            float(data.get('Age'))
        ]

        # Load scaler for diabetes model
        scaler = models['diabetes'].get('scaler')
        
        model = models['diabetes'][selected_model]
        
        if scaler:
            # Apply standard scaling if scaler is available
            features = scaler.transform([features])[0]

        prediction = model.predict([features])
        prediction_proba = model.predict_proba([features])[0]
        
        return jsonify({
            'diabetes_prediction': int(prediction[0]),
            'diabetes_model': selected_model,
            'diabetes_probability': round(max(prediction_proba) * 100, 2)
        })

    elif selected_model in models['heart_disease']:
        # Prepare features dictionary with one-hot encoded features
        input_dict = {
            'Age': float(data.get('Age')),
            'RestingBP': float(data.get('RestingBP')),
            'Cholesterol': float(data.get('Cholesterol')),
            'MaxHR': float(data.get('MaxHR')),
            'Oldpeak': float(data.get('Oldpeak')),
            
            # One-hot encoded columns
            'Sex_M': 1.0 if data.get('Sex') == 'M' else 0.0,
            'ChestPainType_ATA': 1.0 if data.get('ChestPainType') == 'ATA' else 0.0,
            'ChestPainType_NAP': 1.0 if data.get('ChestPainType') == 'NAP' else 0.0,
            'ChestPainType_TA': 1.0 if data.get('ChestPainType') == 'TA' else 0.0,
            'FastingBS_0': 1.0 if data.get('FastingBS') == 0 else 0.0,
            'FastingBS_1': 1.0 if data.get('FastingBS') == 1 else 0.0,
            'RestingECG_Normal': 1.0 if data.get('RestingECG') == 'Normal' else 0.0,
            'RestingECG_ST': 1.0 if data.get('RestingECG') == 'ST' else 0.0,
            'ExerciseAngina_Y': 1.0 if data.get('ExerciseAngina') == 'Y' else 0.0,
            'ST_Slope_Flat': 1.0 if data.get('ST_Slope') == 'Flat' else 0.0,
            'ST_Slope_Up': 1.0 if data.get('ST_Slope') == 'Up' else 0.0
        }

        # Convert to list in the order of training columns
        features = [
            input_dict['Age'],
            input_dict['RestingBP'],
            input_dict['Cholesterol'],
            input_dict['MaxHR'],
            input_dict['Oldpeak'],
            input_dict['Sex_M'],
            input_dict['ChestPainType_ATA'],
            input_dict['ChestPainType_NAP'],
            input_dict['ChestPainType_TA'],
            input_dict['FastingBS_1'],
            input_dict['RestingECG_Normal'],
            input_dict['RestingECG_ST'],
            input_dict['ExerciseAngina_Y'],
            input_dict['ST_Slope_Flat'],
            input_dict['ST_Slope_Up']
        ]

        # Load scaler for heart disease model
        scaler = models['heart_disease'].get('scaler')
        
        model = models['heart_disease'][selected_model]
        
        # Apply standard scaling if scaler is available
        if scaler:
            # Separate numerical and categorical features
            numerical_features = features[:5]
            categorical_features = features[5:]
            
            # Scale only numerical features
            scaled_numerical_features = scaler.transform([numerical_features])[0]
            
            # Recombine scaled numerical features with categorical features
            features = list(scaled_numerical_features) + categorical_features
        
        
        prediction = model.predict([features])
        prediction_proba = model.predict_proba([features])[0]

        # Convert numpy types to standard Python types for JSON serialization
        prediction = int(prediction[0])
        prediction_proba = float(max(prediction_proba))

        return jsonify({
            'heart_disease_prediction': prediction,
            'heart_disease_model': selected_model,
            'heart_disease_probability': round(prediction_proba * 100, 2)
        })
    

    return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
