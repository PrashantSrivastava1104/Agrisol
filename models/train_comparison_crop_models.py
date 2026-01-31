"""
Train comparison models for crop prediction
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load
import xgboost as xgb

from config import (
    CROP_DATA, LABEL_MAPS, LABEL_MAPS_CLASSINDEX, CROP_SCALER_PATH,
    CROP_RF_MODEL_PATH, CROP_XGB_MODEL_PATH, CROP_SVM_MODEL_PATH, 
    CROP_LR_MODEL_PATH, CROP_COMPARISON_METRICS, CROP_MODEL_PATH
)
from models.utils import ensure_label_maps, load_and_scale, train_val_split
import tf_keras as keras


def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train a model and return its metrics"""
    print(f"\nTraining {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'precision': float(val_precision),
        'recall': float(val_recall),
        'f1_score': float(val_f1),
        'training_time_seconds': float(training_time)
    }
    
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    return metrics


def evaluate_neural_network(X_train, y_train, X_val, y_val):
    """Evaluate the existing Neural Network model"""
    print(f"\nEvaluating Neural Network (Primary Model)...")
    
    if not os.path.exists(CROP_MODEL_PATH):
        print("WARNING: Neural Network model not found. Please train it first.")
        return None
    
    model = keras.models.load_model(CROP_MODEL_PATH)
    
    y_pred_train = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'precision': float(val_precision),
        'recall': float(val_recall),
        'f1_score': float(val_f1),
        'training_time_seconds': 0.0
    }
    
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    
    return metrics


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CROP PREDICTION - MODEL COMPARISON TRAINING")
    print("="*60)
    
    print("\nLoading dataset...")
    df = pd.read_csv(CROP_DATA)
    required = ['N','P','K','temperature','humidity','ph','rainfall','label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    
    # Fit label encoder on actual data labels
    le = LabelEncoder()
    le.fit(df['label'].unique())
    
    df_train, df_val = train_val_split(df)
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    
    X_train, _ = load_and_scale(df_train, features, scaler_path=CROP_SCALER_PATH, fit=False)
    X_val, _   = load_and_scale(df_val, features, scaler_path=CROP_SCALER_PATH, fit=False)
    y_train = le.transform(df_train['label'])
    y_val   = le.transform(df_val['label'])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {len(le.classes_)}")
    
    all_metrics = {}
    
    # 1. Evaluate Neural Network
    nn_metrics = evaluate_neural_network(X_train, y_train, X_val, y_val)
    if nn_metrics:
        all_metrics['neural_network'] = nn_metrics
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_metrics = train_and_evaluate_model(rf_model, "Random Forest", X_train, y_train, X_val, y_val)
    dump(rf_model, CROP_RF_MODEL_PATH)
    all_metrics['random_forest'] = rf_metrics
    print(f"Model saved to: {CROP_RF_MODEL_PATH}")
    
    # 3. XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    xgb_metrics = train_and_evaluate_model(xgb_model, "XGBoost", X_train, y_train, X_val, y_val)
    dump(xgb_model, CROP_XGB_MODEL_PATH)
    all_metrics['xgboost'] = xgb_metrics
    print(f"Model saved to: {CROP_XGB_MODEL_PATH}")
    
    # 4. Support Vector Machine
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_metrics = train_and_evaluate_model(svm_model, "Support Vector Machine", X_train, y_train, X_val, y_val)
    dump(svm_model, CROP_SVM_MODEL_PATH)
    all_metrics['svm'] = svm_metrics
    print(f"Model saved to: {CROP_SVM_MODEL_PATH}")
    
    # 5. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_metrics = train_and_evaluate_model(lr_model, "Logistic Regression", X_train, y_train, X_val, y_val)
    dump(lr_model, CROP_LR_MODEL_PATH)
    all_metrics['logistic_regression'] = lr_metrics
    print(f"Model saved to: {CROP_LR_MODEL_PATH}")
    
    # Save metrics
    with open(CROP_COMPARISON_METRICS, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {CROP_COMPARISON_METRICS}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'Time (s)':<12}")
    print("-" * 60)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name.replace('_', ' ').title():<25} "
              f"{metrics['val_accuracy']:.4f}       "
              f"{metrics['f1_score']:.4f}       "
              f"{metrics['training_time_seconds']:.2f}")
    
    best_model = max(all_metrics.items(), key=lambda x: x[1]['val_accuracy'])
    print("\n" + "="*60)
    print(f"BEST Model: {best_model[0].replace('_', ' ').title()}")
    print(f"Accuracy: {best_model[1]['val_accuracy']:.4f}")
    print("="*60)
    print("\nAll crop comparison models trained successfully!\n")
