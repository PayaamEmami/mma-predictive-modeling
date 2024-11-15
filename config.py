# config.py
import os
from datetime import datetime
import torch

# google drive mount path
DRIVE_MOUNT_PATH = '/content/drive'

# base paths
BASE_PATH = os.path.join(DRIVE_MOUNT_PATH, 'MyDrive', 'files')
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUTPUT_BASE_PATH = os.path.join(BASE_PATH, 'models', 'ufc', 'output')

# timestamp for output directory
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH, TIMESTAMP)

# create output and data directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
