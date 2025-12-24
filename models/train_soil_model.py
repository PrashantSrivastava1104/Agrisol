import os


import pandas as pd
import tf_keras as keras
from tf_keras import layers
from config import SOIL_DATA, SOIL_CLF_MODEL_PATH, SOIL_REG_MODEL_PATH, SOIL_SCALER_PATH
from models.utils import load_and_scale, train_val_split
if __name__ == "__main__":
    os.makedirs(os.path.dirname(SOIL_CLF_MODEL_PATH), exist_ok=True)
    df = pd.read_csv(SOIL_DATA)
    rename_map = {
        'Temperature':'temperature','Temp':'temperature',
        'Humidity':'humidity',
        'Soil_Moisture':'moisture','Moisture':'moisture',
        'pH':'ph','PH':'ph',
        'EC':'ec','Electrical_Conductivity':'ec',
        'Nitrogen':'N','N':'N',
        'Phosphorus':'P','P':'P',
        'Potassium':'K','K':'K',
        'Pump':'pump','Irrigation':'pump'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    needed = ['temperature','humidity','ph','ec','N','P','K','moisture','pump']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing soil columns: {missing}")
    feature_cols = ['temperature','humidity','ph','ec','N','P','K']
    df_train, df_val = train_val_split(df)
    X_train, _ = load_and_scale(df_train, feature_cols, scaler_path=SOIL_SCALER_PATH, fit=True)
    X_val, _   = load_and_scale(df_val, feature_cols, scaler_path=SOIL_SCALER_PATH, fit=False)
    y_clf_train = df_train['pump'].astype(int).values
    y_clf_val   = df_val['pump'].astype(int).values
    clf = keras.Sequential([
        layers.Input(shape=(len(feature_cols),)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    clf.fit(X_train, y_clf_train, validation_data=(X_val, y_clf_val), epochs=20, verbose=2)
    clf.save(SOIL_CLF_MODEL_PATH)
    y_reg_train = df_train['moisture'].values
    y_reg_val   = df_val['moisture'].values
    reg = keras.Sequential([
        layers.Input(shape=(len(feature_cols),)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    reg.compile(optimizer='adam', loss='mse', metrics=['mae'])
    reg.fit(X_train, y_reg_train, validation_data=(X_val, y_reg_val), epochs=20, verbose=2)
    reg.save(SOIL_REG_MODEL_PATH)
    print("✅ Soil models trained.")






