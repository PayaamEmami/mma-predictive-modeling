# config.py

import os
from datetime import datetime
import torch

# mount google drive (if using colab)
from google.colab import drive
drive.mount('/content/drive')

# paths
BASE_PATH = '/content/drive/MyDrive/files'
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUTPUT_PATH = os.path.join(BASE_PATH, 'models/ufc/output', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# ensure directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
