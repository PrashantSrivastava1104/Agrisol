import json


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
def ensure_label_maps(path):
    default = {
        "crops": [
            "rice","wheat","maize","chickpea","kidneybeans",
            "pigeonpeas","mothbeans","mungbean","blackgram","lentil",
            "pomegranate","banana","mango","grapes","watermelon",
            "muskmelon","apple","orange","papaya","coconut",
            "cotton","jute","coffee","sugarcane"
        ]
    }
    if not os.path.exists(path):
        with open(path, 'w') as f: json.dump(default, f, indent=2)
        return default
    with open(path, 'r') as f: obj = json.load(f)
    if 'crops' not in obj:
        obj = default
        with open(path, 'w') as f: json.dump(obj, f, indent=2)
    return obj
def load_and_scale(df, feature_cols, target_col=None, scaler_path=None, fit=True):
    scaler = StandardScaler()
    X = df[feature_cols].values.astype('float32')
    if fit:
        Xs = scaler.fit_transform(X)
        if scaler_path: dump(scaler, scaler_path)
    else:
        scaler = load(scaler_path)
        Xs = scaler.transform(X)
    y = df[target_col].values if target_col else None
    return Xs, y
def train_val_split(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)






