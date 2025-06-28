from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.append('/src/rfir/project/aptos2019-blindness-detection')

from aptos.data_loader import ImgProcessor


DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw/diabetic-retinopathy-detection')
TRAIN_DIR = DATA_DIR / 'resized_train_cropped'
TEST_DIR = DATA_DIR / 'resized_test'


NWORKERS = 8
train_imgs = list(TRAIN_DIR.glob('*.jpeg'))
test_imgs = list(TEST_DIR.glob('*.jpg'))
print(len(train_imgs), len(test_imgs))


processor = ImgProcessor()


PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed/diabetic-retinopathy-detection')
PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

def process_and_save(img_path, save_dir):
    """Process a single image and save the result."""
    try:
        img = processor(str(img_path))
        save_as = save_dir / img_path.stem
        np.save(save_as, img)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    futures = [executor.submit(process_and_save, f, PRO_TRAIN_DIR) for f in train_imgs]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

PRO_TEST_DIR = PROCESS_DIR / 'test_images'
PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    futures = [executor.submit(process_and_save, f, PRO_TEST_DIR) for f in test_imgs]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass


train_labels_filename = '/src/rfir/project/aptos2019-blindness-detection/data/raw/diabetic-retinopathy-detection/trainLabels_cropped.csv'
train_df = pd.read_csv(train_labels_filename)[['image', 'level']]
train_df.columns = ['id_code', 'diagnosis']
print(train_df.shape)
train_df.head(2)

test_labels_filename = '/src/rfir/project/aptos2019-blindness-detection/data/raw/diabetic-retinopathy-detection/testLabels.csv'
test_df = pd.read_csv(test_labels_filename)[['image', 'level']]
test_df.columns = ['id_code', 'diagnosis']
print(test_df.shape)
test_df.head(2)

# Save training and test data separately
train_df.to_csv(PROCESS_DIR / 'train.csv', index=False)
test_df.to_csv(PROCESS_DIR / 'test.csv', index=False)

