import os


import pandas as pd
import tf_keras as keras
from tf_keras import layers
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from config import CROP_DATA, LABEL_MAPS, LABEL_MAPS_CLASSINDEX, CROP_MODEL_PATH, CROP_SCALER_PATH
from models.utils import ensure_label_maps, load_and_scale, train_val_split
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CROP_MODEL_PATH), exist_ok=True)
    df = pd.read_csv(CROP_DATA)
    required = ['N','P','K','temperature','humidity','ph','rainfall','label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    label_maps = ensure_label_maps(LABEL_MAPS)
    le = LabelEncoder()
    le.fit(label_maps['crops'])
    df_train, df_val = train_val_split(df)
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    X_train, _ = load_and_scale(df_train, features, scaler_path=CROP_SCALER_PATH, fit=True)
    X_val, _   = load_and_scale(df_val, features, scaler_path=CROP_SCALER_PATH, fit=False)
    y_train = le.transform(df_train['label'])
    y_val   = le.transform(df_val['label'])
    model = keras.Sequential([
        layers.Input(shape=(len(features),)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, verbose=2)
    model.save(CROP_MODEL_PATH)
    dump({int(i): c for i, c in enumerate(le.classes_)}, LABEL_MAPS_CLASSINDEX)
    print("✅ Crop model trained.")






