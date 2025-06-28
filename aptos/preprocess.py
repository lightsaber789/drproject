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

from aptos.data_loader import ImgProcessor


DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw')
TRAIN_DIR = DATA_DIR / 'train_images'
TEST_DIR = DATA_DIR / 'test_images'
NWORKERS = 8

train_imgs = list(TRAIN_DIR.glob('*.png'))
test_imgs = list(TEST_DIR.glob('*.png'))
print(len(train_imgs), len(test_imgs))

processor = ImgProcessor()


PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed')
PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
PRO_TEST_DIR = PROCESS_DIR / 'test_images'

PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)


with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(processor, str(f)): f for f in train_imgs}
    for future in tqdm(as_completed(results), total=len(train_imgs)):
        try:
            img = future.result()
            if img is None:
                continue  # Skip unreadable images
            f = results[future]
            save_as = PRO_TRAIN_DIR / f.stem
            np.save(save_as, img)
        except Exception as e:
            print(f"Error processing {results[future]}: {e}")

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(processor, str(f)): f for f in test_imgs}
    for future in tqdm(as_completed(results), total=len(test_imgs)):
        try:
            img = future.result()
            if img is None:
                continue  # Skip unreadable images
            f = results[future]
            save_as = PRO_TEST_DIR / f.stem
            np.save(save_as, img)
        except Exception as e:
            print(f"Error processing {results[future]}: {e}")


train_csv = 'train.csv'
test_csv = 'test.csv'

shutil.copy(DATA_DIR / train_csv, PROCESS_DIR / train_csv)
shutil.copy(DATA_DIR / test_csv, PROCESS_DIR / test_csv)



pro_train_imgs = list(PRO_TRAIN_DIR.glob('*.npy'))
pro_test_imgs = list(PRO_TEST_DIR.glob('*.npy'))
print(len(pro_train_imgs), len(pro_test_imgs))

means = np.zeros((3, len(pro_train_imgs)))
mean_residuals = np.zeros((3, len(pro_train_imgs)))


def load(filename):
    x = np.load(filename)
    return T.ToTensor()(x)


with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        img = future.result()
        f = results[future]
        idx = pro_train_imgs.index(f)

        # extract mean and variance for nonzero parts of image
        nonzero = (img > 0)
        for c in range(3):
            values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
            means[c, idx] = values.mean()

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        img = future.result()
        f = results[future]
        idx = pro_train_imgs.index(f)

        # extract mean and variance for nonzero parts of image
        nonzero = (img > 0)
        for c in range(3):
            values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
            mean_residuals[c, idx] = ((values - means[c, :].mean()) ** 2).mean()


print(means[0, :].mean())
print(means[1, :].mean())
print(means[2, :].mean())
print(np.sqrt(mean_residuals[0, :].mean()))
print(np.sqrt(mean_residuals[1, :].mean()))
print(np.sqrt(mean_residuals[2, :].mean()))
