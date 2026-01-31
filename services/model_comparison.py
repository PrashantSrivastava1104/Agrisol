"""
Service layer for loading and running comparison models for research analysis
"""
import os
import json
import numpy as np
from joblib import load
import tf_keras as keras

from config import (
    CROP_MODEL_PATH, CROP_RF_MODEL_PATH, CROP_XGB_MODEL_PATH,
    CROP_SVM_MODEL_PATH, CROP_LR_MODEL_PATH, CROP_COMPARISON_METRICS,
    LABEL_MAPS_CLASSINDEX
)


# Cache for loaded models
_comparison_models = {
    'neural_network': None,
    'random_forest': None,
    'xgboost': None,
    'svm': None,
    'logistic_regression': None
}


def load_crop_comparison_models():
    """Load all crop comparison models"""
    global _comparison_models
    
    # Load Neural Network
    if _comparison_models['neural_network'] is None and os.path.exists(CROP_MODEL_PATH):
        _comparison_models['neural_network'] = keras.models.load_model(CROP_MODEL_PATH)
    
    # Load Random Forest
    if _comparison_models['random_forest'] is None and os.path.exists(CROP_RF_MODEL_PATH):
        _comparison_models['random_forest'] = load(CROP_RF_MODEL_PATH)
    
    # Load XGBoost
    if _comparison_models['xgboost'] is None and os.path.exists(CROP_XGB_MODEL_PATH):
        _comparison_models['xgboost'] = load(CROP_XGB_MODEL_PATH)
    
    # Load SVM
    if _comparison_models['svm'] is None and os.path.exists(CROP_SVM_MODEL_PATH):
        _comparison_models['svm'] = load(CROP_SVM_MODEL_PATH)
    
    # Load Logistic Regression
    if _comparison_models['logistic_regression'] is None and os.path.exists(CROP_LR_MODEL_PATH):
        _comparison_models['logistic_regression'] = load(CROP_LR_MODEL_PATH)
    
    return _comparison_models


def predict_crop_all_models(input_scaled, class_map):
    """
    Run all models on the input and return predictions
    
    Args:
        input_scaled: Scaled input features (numpy array)
        class_map: Dictionary mapping class indices to crop names
        
    Returns:
        Dictionary with predictions from each model
    """
    models = load_crop_comparison_models()
    predictions = {}
    
    # Neural Network
    if models['neural_network'] is not None:
        probs = models['neural_network'].predict(input_scaled, verbose=0)[0]
        top_idx = np.argmax(probs)
        predictions['neural_network'] = {
            'crop': class_map[int(top_idx)],
            'confidence': float(probs[top_idx])
        }
    
    # Random Forest
    if models['random_forest'] is not None:
        pred_idx = models['random_forest'].predict(input_scaled)[0]
        probs = models['random_forest'].predict_proba(input_scaled)[0]
        predictions['random_forest'] = {
            'crop': class_map[int(pred_idx)],
            'confidence': float(probs[pred_idx])
        }
    
    # XGBoost
    if models['xgboost'] is not None:
        pred_idx = models['xgboost'].predict(input_scaled)[0]
        probs = models['xgboost'].predict_proba(input_scaled)[0]
        predictions['xgboost'] = {
            'crop': class_map[int(pred_idx)],
            'confidence': float(probs[pred_idx])
        }
    
    # SVM
    if models['svm'] is not None:
        pred_idx = models['svm'].predict(input_scaled)[0]
        # SVM doesn't have predict_proba by default, use decision_function
        if hasattr(models['svm'], 'predict_proba'):
            probs = models['svm'].predict_proba(input_scaled)[0]
            predictions['svm'] = {
                'crop': class_map[int(pred_idx)],
                'confidence': float(probs[pred_idx])
            }
        else:
            predictions['svm'] = {
                'crop': class_map[int(pred_idx)],
                'confidence': 0.0  # SVM without probability
            }
    
    # Logistic Regression
    if models['logistic_regression'] is not None:
        pred_idx = models['logistic_regression'].predict(input_scaled)[0]
        probs = models['logistic_regression'].predict_proba(input_scaled)[0]
        predictions['logistic_regression'] = {
            'crop': class_map[int(pred_idx)],
            'confidence': float(probs[pred_idx])
        }
    
    return predictions


def get_crop_metrics():
    """Load saved metrics from JSON"""
    if os.path.exists(CROP_COMPARISON_METRICS):
        with open(CROP_COMPARISON_METRICS, 'r') as f:
            return json.load(f)
    return {}


def format_model_name(model_key):
    """Format model key to display name"""
    names = {
        'neural_network': 'Neural Network',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'svm': 'Support Vector Machine',
        'logistic_regression': 'Logistic Regression'
    }
    return names.get(model_key, model_key.replace('_', ' ').title())
