import os


from flask import Flask, render_template, request, jsonify
import numpy as np
from joblib import load
import tf_keras as keras
from flask_login import LoginManager, login_required, current_user
from config import (
    SECRET_KEY,
    CROP_SCALER_PATH, SOIL_SCALER_PATH,
    CROP_MODEL_PATH, SOIL_CLF_MODEL_PATH, SOIL_REG_MODEL_PATH,
    LABEL_MAPS_CLASSINDEX,
    SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS,
)
from services.sensor_ingest import append_soil_reading
from services.model_comparison import predict_crop_all_models, get_crop_metrics
from database import db, User, CropPrediction, SoilPrediction
from auth import auth_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register authentication blueprint
app.register_blueprint(auth_bp)

# Create database tables
with app.app_context():
    db.create_all()


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

@app.route('/tools/crop')
@login_required
def crop_prediction_page():
    """Crop prediction tool page"""
    return render_template('crop_prediction.html')

@app.route('/tools/soil')
@login_required
def soil_monitoring_page():
    """Soil monitoring tool page"""
    return render_template('soil_monitoring.html')

@app.route('/history')
@login_required
def history():
    """Display user's prediction history"""
    import json
    
    # Get page number from query params (default to 1)
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Get crop predictions for current user
    crop_predictions = CropPrediction.query.filter_by(user_id=current_user.id)\
        .order_by(CropPrediction.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    # Get soil predictions for current user
    soil_predictions = SoilPrediction.query.filter_by(user_id=current_user.id)\
        .order_by(SoilPrediction.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    # Parse JSON for soil predictions
    for pred in soil_predictions.items:
        if pred.recommended_crops:
            try:
                pred.recommended_crops_list = json.loads(pred.recommended_crops)
            except:
                pred.recommended_crops_list = []
        else:
            pred.recommended_crops_list = []
    
    return render_template('history.html', 
                         crop_predictions=crop_predictions,
                         soil_predictions=soil_predictions)


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
@login_required
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
    
    # Load class map
    class_map = load(LABEL_MAPS_CLASSINDEX)
    
    # Create list of all crops with their confidence scores
    all_crops = []
    for idx, prob in enumerate(probs):
        crop_name = class_map[int(idx)]
        adjusted_confidence = float(prob) * similarity_score
        all_crops.append({
            'name': crop_name,
            'confidence': adjusted_confidence,
            'raw_confidence': float(prob)
        })
    
    # Sort by confidence descending
    all_crops.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get top prediction
    top_1_crop = all_crops[0]['name']
    confidence = all_crops[0]['confidence']
    
    # Check if top 2 are very close (within 1%)
    if len(all_crops) > 1 and (all_crops[0]['raw_confidence'] - all_crops[1]['raw_confidence']) <= 0.01:
        pred_label = f"{all_crops[0]['name']}/{all_crops[1]['name']}"
    else:
        pred_label = top_1_crop
    
    # Save prediction to database
    try:
        crop_prediction = CropPrediction(
            user_id=current_user.id,
            N=N,
            P=P,
            K=K,
            temperature=temp,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            predicted_crop=pred_label,
            confidence=confidence
        )
        db.session.add(crop_prediction)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving crop prediction: {e}")
    
    # Get predictions from all comparison models
    model_predictions = {}
    model_metrics = {}
    try:
        model_predictions = predict_crop_all_models(Xs, class_map)
        model_metrics = get_crop_metrics()
    except Exception as e:
        print(f"Model comparison failed: {e}")
    
    result = {
        'predicted_crop': pred_label,
        'confidence': confidence,
        'all_crops': all_crops,
        'model_predictions': model_predictions,
        'model_metrics': model_metrics
    }
    return render_template('results.html', result=result)



@app.route('/predict/soil', methods=['POST'])
@login_required
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
    
    # Save prediction to database
    import json
    try:
        soil_prediction = SoilPrediction(
            user_id=current_user.id,
            temperature=temp,
            humidity=humidity,
            ph=ph,
            ec=ec,
            N=N,
            P=P,
            K=K,
            irrigation_needed=bool(irr_need),
            predicted_moisture=moisture_pred,
            recommended_crops=json.dumps(recommended_crops)
        )
        db.session.add(soil_prediction)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving soil prediction: {e}")
    
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






