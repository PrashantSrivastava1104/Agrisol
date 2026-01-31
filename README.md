# AgriSol

A Flask-based ML project for crop recommendation and soil monitoring using TensorFlow and scikit-learn. Models are trained from Kaggle datasets.

## Windows Setup

- Prerequisites:
  - Python 3.10 or 3.11 (x64) installed and added to PATH
  - Git (optional but recommended)
  - A Kaggle account

### 1) Create and activate a virtual environment

```powershell
cd C:\\Users\\prash\\OneDrive\\Desktop\\AgriSol\\AgriSol
python -m venv .venv
. .venv\\Scripts\\Activate.ps1
```

To deactivate later:
```powershell
deactivate
```

### 2) Install Python packages

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow fails to install on CPU-only machines, ensure you’re on Python 3.10/3.11 and Windows 10/11 with AVX support.

### 3) Configure environment variables

Copy the example env file and set a secure secret key if desired:
```powershell
Copy-Item .env.example .env
# Edit .env if you want to change SECRET_KEY
```

### 4) Configure Kaggle API (kaggle.json)

- Go to Kaggle Account settings → Create New API Token. This downloads `kaggle.json`.
- Place it as:
  - `%USERPROFILE%\\.kaggle\\kaggle.json` (recommended), or
  - In this project at `data\\kaggle\\kaggle.json`.

If using project-local file, set environment variables (optional):
```powershell
$env:KAGGLE_CONFIG_DIR = (Resolve-Path .\\data\\kaggle).Path
```

### 5) Download datasets

This project expects the following CSVs:
- Crop dataset → `data\\kaggle\\Crop_recommendation.csv`
- Soil/Irrigation dataset → `data\\kaggle\\AutoIrrigation.csv`

Download via Kaggle CLI (two examples; adjust if your dataset names differ):
```powershell
# Ensure kaggle CLI sees your credentials
kaggle datasets list | Select-Object -First 5

# Example: Crop Recommendation
kaggle datasets download -d atharvaingle/crop-recommendation-dataset
Expand-Archive .\\crop-recommendation-dataset.zip -DestinationPath .\\data\\kaggle -Force
Remove-Item .\\crop-recommendation-dataset.zip

# Example: Auto Irrigation (replace with the dataset you choose)
# Search Kaggle for a dataset containing pump/moisture + N,P,K,EC,pH,temperature,humidity
# Suppose file extracted is AutoIrrigation.csv
# Place the CSV at: .\\data\\kaggle\\AutoIrrigation.csv
```

Confirm file paths:
```powershell
Get-Item .\\data\\kaggle\\Crop_recommendation.csv
Get-Item .\\data\\kaggle\\AutoIrrigation.csv
```

### 6) Train ML models

- **[New]** Crop Comparison Models (RF, XGB, SVM, LR):
```powershell
python .\\models\\train_comparison_crop_models.py
```
Artifacts produced:
- `models\\crop_random_forest.pkl`
- `models\\crop_xgboost.pkl`
- `models\\crop_svm.pkl`
- `models\\crop_logistic_regression.pkl`
- `models\\crop_comparison_metrics.json`

- Crop model (Primary Neural Network):
```powershell
python .\\models\\train_crop_model.py
```
Artifacts produced:
- `models\\crop_model.keras`
- `models\\crop_scaler.pkl`
- `data\\label_maps_classindex.pkl`
- `data\\label_maps.json` (auto-created if missing)

- Soil models (classification for irrigation + regression for moisture):
```powershell
python .\\models\\train_soil_model.py
```
Artifacts produced:
- `models\\soil_clf_model.keras`
- `models\\soil_reg_model.keras`
- `models\\soil_scaler.pkl`

## Key Features

### 1. Crop Prediction & Comparison
- **Primary Recommendation**: Uses a deep learning Neural Network for high accuracy.
- **Crop Comparison**: Dropdown to compare the recommended crop with other potential crops for the same soil conditions.
- **Collapsible Rankings**: Complete list of all 22 crops ranked by suitability confidence.

### 2. Model Comparison (Research Analysis)
- Compares predictions from 5 different ML algorithms:
  - Neural Network (Primary)
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Logistic Regression
- **Visual Charts**: Interactive Bar and Radar charts showing confidence scores and performance metrics (Accuracy, Precision, Recall, F1).

### 3. Soil Monitoring
- Predicts if irrigation is needed based on soil sensors.
- Estimates current soil moisture levels.

### 7) Run Flask app

```powershell
$env:FLASK_APP = 'app.py'
python app.py
```
The app will start at `http://127.0.0.1:5000/`.

### 8) API Endpoints

- `POST /predict/crop`
  - JSON/Form fields: `N,P,K,temperature,humidity,ph,rainfall`
  - Returns: `predicted_crop`, `confidence`

- `POST /predict/soil`
  - JSON/Form fields: `temperature,humidity,ph,ec,N,P,K`
  - Returns: `irrigation_needed`, `irrigation_probability`, `predicted_moisture`

- `POST /api/ingest`
  - JSON body (fields optional): `temperature,humidity,ph,ec,N,P,K`
  - Appends a row to `storage\\logs\\soil_readings.csv`

## Notes

- Directories `data`, `models`, and `storage\\logs` are created automatically by the app/config at runtime.
- Ensure models are trained before calling prediction endpoints, or you will see file-not-found errors when loading models/scalers.
- If you encounter permission issues with PowerShell execution policy, run PowerShell as Administrator and:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```






