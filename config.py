import os


from dotenv import load_dotenv
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
LOG_DIR = os.path.join(STORAGE_DIR, 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
CROP_DATA = os.path.join(DATA_DIR, 'kaggle', 'Crop_recommendation.csv')
SOIL_DATA = os.path.join(DATA_DIR, 'kaggle', 'AutoIrrigation.csv')
LABEL_MAPS = os.path.join(DATA_DIR, 'label_maps.json')
LABEL_MAPS_CLASSINDEX = os.path.join(DATA_DIR, 'label_maps_classindex.pkl')
CROP_MODEL_PATH = os.path.join(MODELS_DIR, 'crop_model.keras')
SOIL_CLF_MODEL_PATH = os.path.join(MODELS_DIR, 'soil_clf_model.keras')
SOIL_REG_MODEL_PATH = os.path.join(MODELS_DIR, 'soil_reg_model.keras')
CROP_SCALER_PATH = os.path.join(MODELS_DIR, 'crop_scaler.pkl')
SOIL_SCALER_PATH = os.path.join(MODELS_DIR, 'soil_scaler.pkl')
SOIL_READINGS_LOG = os.path.join(LOG_DIR, 'soil_readings.csv')
APP_LOG = os.path.join(LOG_DIR, 'app.log')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret')






