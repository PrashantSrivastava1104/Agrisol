import os


from flask import Flask, render_template, request, jsonify
import numpy as np
from joblib import load
import tf_keras as keras
from config import (
    SECRET_KEY,
    CROP_SCALER_PATH, SOIL_SCALER_PATH,
    CROP_MODEL_PATH, SOIL_CLF_MODEL_PATH, SOIL_REG_MODEL_PATH,
    LABEL_MAPS_CLASSINDEX,
)
from services.sensor_ingest import append_soil_reading

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY


_models = {'crop': None, 'soil_clf': None, 'soil_reg': None}
_scalers = {'crop': None, 'soil': None}

def get_crop_model():
    if _models['crop'] is None:
        if not os.path.exists(CROP_MODEL_PATH):
            raise FileNotFoundError(f"Crop model not found at {CROP_MODEL_PATH}. Please train the model first.")
        _models['crop'] = keras.models.load_model(CROP_MODEL_PATH)
    return _models['crop']

def get_soil_models():
    if _models['soil_clf'] is None:
        if not os.path.exists(SOIL_CLF_MODEL_PATH):
            raise FileNotFoundError(f"Soil classification model not found at {SOIL_CLF_MODEL_PATH}. Please train the soil model first.")
        _models['soil_clf'] = keras.models.load_model(SOIL_CLF_MODEL_PATH)
    if _models['soil_reg'] is None:
        if not os.path.exists(SOIL_REG_MODEL_PATH):
            raise FileNotFoundError(f"Soil regression model not found at {SOIL_REG_MODEL_PATH}. Please train the soil model first.")
        _models['soil_reg'] = keras.models.load_model(SOIL_REG_MODEL_PATH)
    return _models['soil_clf'], _models['soil_reg']

def get_crop_scaler():
    if _scalers['crop'] is None:
        if not os.path.exists(CROP_SCALER_PATH):
            raise FileNotFoundError(f"Crop scaler not found at {CROP_SCALER_PATH}. Please train the model first.")
        _scalers['crop'] = load(CROP_SCALER_PATH)
    return _scalers['crop']

def get_soil_scaler():
    if _scalers['soil'] is None:
        if not os.path.exists(SOIL_SCALER_PATH):
            raise FileNotFoundError(f"Soil scaler not found at {SOIL_SCALER_PATH}. Please train the soil model first.")
        _scalers['soil'] = load(SOIL_SCALER_PATH)
    return _scalers['soil']


@app.route('/')
def home():
    return render_template('index.html')

def calculate_input_similarity(N, P, K, temp, humidity, ph, rainfall):
    """
    Calculate how similar the input values are to actual crops in the dataset.
    Returns a confidence multiplier (0.0 to 1.0) based on distance from real data.
    
    If inputs are far from any real crop data, confidence is reduced.
    """
    import pandas as pd
    from config import CROP_DATA
    
    try:
        # Load the actual dataset
        df = pd.read_csv(CROP_DATA)
        
        # Create input array
        input_vals = np.array([N, P, K, temp, humidity, ph, rainfall])
        
        # Calculate normalized distance to each row in dataset
        # Normalize by dataset std to handle different scales
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        dataset_vals = df[feature_cols].values
        
        # Calculate mean and std for normalization
        means = df[feature_cols].mean().values
        stds = df[feature_cols].std().values
        stds[stds == 0] = 1  # Avoid division by zero
        
        # Normalize input and dataset
        input_normalized = (input_vals - means) / stds
        dataset_normalized = (dataset_vals - means) / stds
        
        # Calculate Euclidean distance to each dataset row
        distances = np.sqrt(np.sum((dataset_normalized - input_normalized)**2, axis=1))
        
        # Find minimum distance (closest match)
        min_distance = np.min(distances)
        
        # Convert distance to similarity score (0 to 1)
        # Distance of 0 = perfect match = 1.0 confidence
        # Distance of 5+ = very far = 0.3 confidence (low)
        # Use exponential decay for smooth transition
        similarity = np.exp(-min_distance / 3.0)  # Decay factor of 3
        
        # Ensure minimum confidence of 0.3 (30%) and max of 1.0
        similarity = max(0.3, min(1.0, similarity))
        
        return similarity
        
    except Exception as e:
        # If anything fails, return 1.0 (don't modify confidence)
        print(f"Similarity calculation failed: {e}")
        return 1.0

@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    model = get_crop_model()
    scaler = get_crop_scaler()
    feature_cols = ['N','P','K','temperature','humidity','ph','rainfall']
    if request.is_json:
        data = request.get_json()
        vals = [float(data[c]) for c in feature_cols]
    else:
        vals = [float(request.form[c]) for c in feature_cols]
    
    # Extract input values and calculate similarity to dataset
    N, P, K, temp, humidity, ph, rainfall = vals
    similarity_score = calculate_input_similarity(N, P, K, temp, humidity, ph, rainfall)
    
    X = np.array(vals).reshape(1, -1)
    Xs = scaler.transform(X)
    probs = model.predict(Xs, verbose=0)[0]
    
    # Get top 2 predictions
    top_2_indices = np.argsort(probs)[-2:][::-1]  # Get indices of top 2, sorted descending
    top_1_prob = float(probs[top_2_indices[0]])
    top_2_prob = float(probs[top_2_indices[1]])
    
    class_map = load(LABEL_MAPS_CLASSINDEX)
    top_1_crop = class_map[int(top_2_indices[0])]
    top_2_crop = class_map[int(top_2_indices[1])]
    
    # If top 2 crops are within 1% confidence, show both
    if top_1_prob - top_2_prob <= 0.01:
        pred_label = f"{top_1_crop}/{top_2_crop}"
        confidence = top_1_prob  # Use top prediction's confidence
    else:
        pred_label = top_1_crop
        confidence = top_1_prob
    
    # Adjust confidence based on similarity to real dataset
    # If inputs are far from any real crop data, reduce confidence
    confidence = confidence * similarity_score
    
    result = {
        'predicted_crop': pred_label,
        'confidence': confidence
    }
    return render_template('results.html', result=result)



@app.route('/predict/soil', methods=['POST'])
def predict_soil():
    import pandas as pd
    from config import CROP_DATA
    
    soil_clf, soil_reg = get_soil_models()
    scaler = get_soil_scaler()
    feature_cols = ['temperature','humidity','ph','ec','N','P','K']
    if request.is_json:
        data = request.get_json()
        vals = [float(data[c]) for c in feature_cols]
    else:
        vals = [float(request.form[c]) for c in feature_cols]
    X = np.array(vals).reshape(1, -1)
    Xs = scaler.transform(X)
    
    # Get predictions from both models
    irr_prob = float(soil_clf.predict(Xs, verbose=0)[0][0])
    moisture_pred = float(soil_reg.predict(Xs, verbose=0)[0][0])
    
    # Use moisture-based logic for more accurate irrigation decision
    temp, humidity, ph, ec, N, P, K = vals
    
    # Primary logic: moisture threshold
    if moisture_pred < 45:
        irr_need = 1
    # Secondary logic: extreme conditions (hot and dry)
    elif temp > 33 and humidity < 50:
        irr_need = 1
    else:
        irr_need = 0
    
    # Recommend crops based on soil conditions using the trained model
    try:
        # Use the actual crop prediction model to get recommendations
        model = get_crop_model()
        crop_scaler = get_crop_scaler()
        
        # We have: temp, humidity, pH, N, P, K from soil monitoring
        # Crop model needs: N, P, K, temperature, humidity, pH, rainfall
        # We'll test with different rainfall values to find suitable crops
        
        recommended_crops = []
        crop_predictions = {}
        
        # Test with typical rainfall values (50, 100, 150, 200mm)
        rainfall_values = [50, 100, 150, 200]
        
        for rainfall in rainfall_values:
            # Prepare input: N, P, K, temp, humidity, pH, rainfall
            X_test = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            Xs_test = crop_scaler.transform(X_test)
            probs = model.predict(Xs_test, verbose=0)[0]
            
            # Get top prediction
            top_idx = np.argmax(probs)
            top_prob = float(probs[top_idx])
            
            class_map = load(LABEL_MAPS_CLASSINDEX)
            crop_name = class_map[int(top_idx)].capitalize()
            
            # Store the best confidence for each crop
            if crop_name not in crop_predictions or top_prob > crop_predictions[crop_name]:
                crop_predictions[crop_name] = top_prob
        
        # Filter crops with confidence >= 60% and sort by confidence
        MIN_CONFIDENCE = 0.60
        suitable_crops = [(crop, conf * 100) for crop, conf in crop_predictions.items() if conf >= MIN_CONFIDENCE]
        suitable_crops.sort(key=lambda x: x[1], reverse=True)
        
        # Get top crop
        if suitable_crops:
            top_crop, top_conf = suitable_crops[0]
            recommended_crops.append({
                'name': top_crop,
                'suitability': round(top_conf, 1)
            })
            
            # Add second crop if within 5% and confidence >= 60%
            if len(suitable_crops) > 1:
                second_crop, second_conf = suitable_crops[1]
                if abs(top_conf - second_conf) <= 5:
                    recommended_crops.append({
                        'name': second_crop,
                        'suitability': round(second_conf, 1)
                    })
        
    except Exception as e:
        print(f"Crop recommendation failed: {e}")
        recommended_crops = []
    
    result = {
        'irrigation_needed': irr_need,
        'irrigation_probability': irr_prob,
        'predicted_moisture': moisture_pred,
        'recommended_crops': recommended_crops
    }
    return render_template('results.html', result=result)


@app.route('/api/ingest', methods=['POST'])
def ingest_sensor():
    data = request.get_json()
    append_soil_reading(data)
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(debug=True)






