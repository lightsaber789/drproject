from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import torchvision.transforms as T

# Add the project root to the Python path
import sys
sys.path.append('/src/rfir/project/aptos2019-blindness-detection')

from aptos.data_loader import ImgProcessor

# Function to convert an image to grayscale
def convert_to_grayscale(img):
    """Convert an image to grayscale using OpenCV."""
    if len(img.shape) == 3 and img.shape[2] == 3:  # Check if the image is RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img  # If already grayscale, return as is

# Function to process and save an image
def process_and_save(img_path, save_dir, processor):
    """Process a single image and save the result."""
    try:
        img = processor(str(img_path))
        img = convert_to_grayscale(img)  # Convert to grayscale
        save_as = save_dir / img_path.stem
        np.save(save_as, img)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# Preprocess Dataset 1
def preprocess_dataset_1():
    DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw')
    TRAIN_DIR = DATA_DIR / 'train_images'
    TEST_DIR = DATA_DIR / 'test_images'
    NWORKERS = 8

    train_imgs = list(TRAIN_DIR.glob('*.png'))
    test_imgs = list(TEST_DIR.glob('*.png'))
    print(f"Dataset 1: Found {len(train_imgs)} training images and {len(test_imgs)} test images.")

    processor = ImgProcessor()

    PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed/greyscale')
    PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
    PRO_TEST_DIR = PROCESS_DIR / 'test_images'

    PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Process training images
    with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        futures = [executor.submit(process_and_save, f, PRO_TRAIN_DIR, processor) for f in train_imgs]
        for _ in tqdm(as_completed(futures), total=len(train_imgs)):
            pass

    # Process test images
    with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        futures = [executor.submit(process_and_save, f, PRO_TEST_DIR, processor) for f in test_imgs]
        for _ in tqdm(as_completed(futures), total=len(test_imgs)):
            pass

    # Copy label files
    train_csv = 'train.csv'
    test_csv = 'test.csv'
    shutil.copy(DATA_DIR / train_csv, PROCESS_DIR / train_csv)
    shutil.copy(DATA_DIR / test_csv, PROCESS_DIR / test_csv)

    print(f"Dataset 1 preprocessing complete. Files saved in {PROCESS_DIR}.")

# Preprocess Dataset 2
def preprocess_dataset_2():
    DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw/diabetic-retinopathy-detection')
    TRAIN_DIR = DATA_DIR / 'resized_train_cropped'
    TEST_DIR = DATA_DIR / 'resized_test'
    NWORKERS = 8

    train_imgs = list(TRAIN_DIR.glob('*.jpeg'))
    test_imgs = list(TEST_DIR.glob('*.jpg'))
    print(f"Dataset 2: Found {len(train_imgs)} training images and {len(test_imgs)} test images.")

    processor = ImgProcessor()

    PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed/greyscale/diabetic-retinopathy-detection')
    PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
    PRO_TEST_DIR = PROCESS_DIR / 'test_images'

    PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Process training images
    with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        futures = [executor.submit(process_and_save, f, PRO_TRAIN_DIR, processor) for f in train_imgs]
        for _ in tqdm(as_completed(futures), total=len(train_imgs)):
            pass

    # Process test images
    with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
        futures = [executor.submit(process_and_save, f, PRO_TEST_DIR, processor) for f in test_imgs]
        for _ in tqdm(as_completed(futures), total=len(test_imgs)):
            pass

    # Load and save labels
    train_labels_filename = DATA_DIR / 'trainLabels_cropped.csv'
    test_labels_filename = DATA_DIR / 'testLabels.csv'

    train_df = pd.read_csv(train_labels_filename)[['image', 'level']]
    train_df.columns = ['id_code', 'diagnosis']
    test_df = pd.read_csv(test_labels_filename)[['image', 'level']]
    test_df.columns = ['id_code', 'diagnosis']

    # Save training and test labels
    train_df.to_csv(PROCESS_DIR / 'train.csv', index=False)
    test_df.to_csv(PROCESS_DIR / 'test.csv', index=False)

    print(f"Dataset 2 preprocessing complete. Files saved in {PROCESS_DIR}.")

# Main function to preprocess both datasets
def main():
    print("Preprocessing Dataset Aptos Greyscale...")
    preprocess_dataset_1()

    print("\nPreprocessing Dataset DR Greyscale...")
    preprocess_dataset_2()

    print("\nAll datasets preprocessed successfully!")

if __name__ == "__main__":
    main()