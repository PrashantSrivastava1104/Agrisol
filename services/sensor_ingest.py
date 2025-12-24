import os


import csv
from datetime import datetime
from config import SOIL_READINGS_LOG
def append_soil_reading(payload: dict):
    os.makedirs(os.path.dirname(SOIL_READINGS_LOG), exist_ok=True)
    is_new = not os.path.exists(SOIL_READINGS_LOG)
    with open(SOIL_READINGS_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['timestamp','temperature','humidity','ph','ec','N','P','K'])
        writer.writerow([
            datetime.utcnow().isoformat(),
            payload.get('temperature'),
            payload.get('humidity'),
            payload.get('ph'),
            payload.get('ec'),
            payload.get('N'),
            payload.get('P'),
            payload.get('K'),
        ])






