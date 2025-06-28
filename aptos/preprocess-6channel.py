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
import torch
import torchvision.transforms as T
import sys
from datetime import datetime, date, time
sys.path.append('/src/rfir/project/aptos2019-blindness-detection')

from aptos.data_loader import ImgProcessor


DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw')
TRAIN_DIR = DATA_DIR / 'train_images'
TEST_DIR = DATA_DIR / 'test_images'
NWORKERS = 2

train_imgs = list(TRAIN_DIR.glob('*.png'))
test_imgs = list(TEST_DIR.glob('*.png'))
print(len(train_imgs), len(test_imgs))

processor = ImgProcessor()
'''
def convert_to_6ch_from_rgb(img):
    if img.shape[0] == 3:  # Already (3, H, W)
        R, G, B = img[0], img[1], img[2]
    else:  # Maybe (H, W, 3), just in case
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    rgb = np.stack([R, G, B], axis=-1).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    RG = (R + G) / 2
    GB = (G + B) / 2
    RB = (R + B) / 2

    HS = (H + S) / 2
    HV = (H + V) / 2
    SV = (S + V) / 2

    return np.stack([RG, GB, RB, HS, HV, SV], axis=0).astype(np.float32)
'''


def convert_to_6ch_from_rgb(img):
    """Convert RGB to 6 channels with  exact combinations"""
    # Get channels in float32
    if img.shape[0] == 3:  # (C,H,W)
        R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
    else:  # (H,W,C)
        R, G, B = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)

    # Add epsilon to avoid division by zero
    eps = 1e-7

    # 1. original weighted RGB combinations
    RG = (0.6 * R + 0.4 * G).clip(0, 255)
    RB = (0.8 * R + 0.2 * B).clip(0, 255)
    GB = (0.3 * G + 0.7 * B).clip(0, 255)

    # 2. Convert to HSV (needed for last 3 channels)
    rgb_uint8 = np.stack([R.clip(0, 255), G.clip(0, 255), B.clip(0, 255)], axis=-1).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)

    # 3.  HSV-based combinations (normalized to comparable ranges)
    HS = ((H / 180.0) * 0.7 + (S / 255.0) * 0.3).clip(0, 1)  # Hue-Saturation blend
    SV = ((S / 255.0) * 0.4 + (V / 255.0) * 0.6).clip(0, 1)  # Saturation-Value blend
    HV = ((H / 180.0) * 0.5 + (V / 255.0) * 0.5).clip(0, 1)  # Hue-Value blend


    return np.stack([
        RG / 255.0,  # Channel 0: Your RG mix
        RB / 255.0,  # Channel 1: Your RB mix
        GB / 255.0,  # Channel 2: Your GB mix
        HS,  # Channel 3: Hue-Saturation
        SV,  # Channel 4: Saturation-Value
        HV  # Channel 5: Hue-Value
    ], axis=0).astype(np.float32)

def concat_6channel(img, input_range=(0, 255)):
    """
    Args:
        img: Input image (H,W,3) or (3,H,W) in [input_range[0], input_range[1]].
        input_range: Tuple (min, max) of input pixel range (e.g., (0, 255) or (0, 1)).
    Returns:
        torch.Tensor: 6-channel (6,H,W) in [0, 1].
    """
    # Convert to (H,W,3) if needed
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    # Normalize to [0, 255] for OpenCV HSV conversion
    min_val, max_val = input_range
    img_uint8 = ((img - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)
    # Convert to HSV

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Concatenate and normalize
    combined = np.concatenate([img, hsv], axis=-1).astype(np.float32)
    combined[..., 0:3] = (combined[..., 0:3] - min_val) / (max_val - min_val)  # RGB
    combined[..., 3] /= 180.0  # H
    combined[..., 4:] /= 255.0  # S and V

    return torch.from_numpy(combined.transpose(2, 0, 1))  # (6,H,W)

def concat_6channel_WEIGHTS(img, input_range=(0, 255), weights=None):
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    min_val, max_val = input_range
    img_uint8 = ((img - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    combined = np.concatenate([img, hsv], axis=-1).astype(np.float32)
    combined[..., 0:3] = (combined[..., 0:3] - min_val) / (max_val - min_val)  # Normalize RGB
    combined[..., 3] /= 180.0  # Normalize H
    combined[..., 4:] /= 255.0  # Normalize S and V

    if weights is not None:
        weights = np.array(weights, dtype=np.float32)
        rgb_weights = weights[:3]
        hsv_weights = weights[3:]

        rgb_weights /= rgb_weights.sum() if rgb_weights.sum() != 0 else 1
        hsv_weights /= hsv_weights.sum() if hsv_weights.sum() != 0 else 1
        weights = np.concatenate([rgb_weights, hsv_weights])

        for i in range(6):
            combined[..., i] *= weights[i]

    return torch.from_numpy(combined.transpose(2, 0, 1))  # (6,H,W)

def concat_6channel_WEIGHTS_with_CLAHE_YCBCR(img, input_range=(0, 255), weights=None):
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

    min_val, max_val = input_range

    # Convert to uint8 for CLAHE
    img_uint8 = ((img - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)

    # --- Apply CLAHE to each RGB channel ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img_uint8[..., i] = clahe.apply(img_uint8[..., i])

    # Convert enhanced RGB to YCBCR
    ycbcr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)

    # Stack RGB and YCBCR
    combined = np.concatenate([img_uint8, ycbcr], axis=-1).astype(np.float32)

#Normalize
    combined[..., 0:3] /= 255.0  # R, G, B
    combined[..., 3] /= 255.0  # Y is 0–255
    combined[..., 4] = (combined[..., 4] - 128) / 127.0  # Cr centered at 128
    combined[..., 5] = (combined[..., 5] - 128) / 127.0  # Cb centered at 128

    # Apply weights if provided
    if weights is not None:
        weights = np.array(weights, dtype=np.float32)
        rgb_weights = weights[:3]
        ycbcr_weights = weights[3:]

        # Normalize weights within each group
        rgb_weights /= rgb_weights.sum() if rgb_weights.sum() != 0 else 1
        ycbcr_weights /= ycbcr_weights.sum() if ycbcr_weights.sum() != 0 else 1
        weights = np.concatenate([rgb_weights, ycbcr_weights])

        for i in range(6):
            combined[..., i] *= weights[i]

    # Return in (C, H, W) format for PyTorch
    return torch.from_numpy(combined.transpose(2, 0, 1))  # shape: (6, H, W)

def concat_6channel_WEIGHTS_with_CLAHE(img, input_range=(0, 255), weights=None):
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

    min_val, max_val = input_range

    # Convert to uint8 for CLAHE
    img_uint8 = ((img - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)

    # --- Apply CLAHE to each RGB channel ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img_uint8[..., i] = clahe.apply(img_uint8[..., i])

    # Convert enhanced RGB to HSV
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Stack RGB and HSV
    combined = np.concatenate([img_uint8, hsv], axis=-1).astype(np.float32)

    # Normalize:
    combined[..., 0:3] /= 255.0            # R, G, B
    combined[..., 3] /= 180.0              # H
    combined[..., 4:] /= 255.0             # S, V

    # Apply weights if provided
    if weights is not None:
        weights = np.array(weights, dtype=np.float32)
        rgb_weights = weights[:3]
        hsv_weights = weights[3:]

        # Normalize weights within each group
        rgb_weights /= rgb_weights.sum() if rgb_weights.sum() != 0 else 1
        hsv_weights /= hsv_weights.sum() if hsv_weights.sum() != 0 else 1
        weights = np.concatenate([rgb_weights, hsv_weights])

        for i in range(6):
            combined[..., i] *= weights[i]

    # Return in (C, H, W) format for PyTorch
    return torch.from_numpy(combined.transpose(2, 0, 1))  # shape: (6, H, W)


def concat_6channel_model_fusion(img, input_range=(0, 255)):
    """Returns tensor in format [R, G, B, H_sin, H_cos, S, V]"""
    #recommended by perplexity ai
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    min_val, max_val = input_range
    img_uint8 = ((img - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)

    # Convert to HSV and process hue circularly
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h_rad = (hsv[..., 0] / 180.0) * np.pi
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)

    # Normalize all components
    rgb_norm = (img - min_val) / (max_val - min_val)
    s_norm = hsv[..., 1] / 255.0
    v_norm = hsv[..., 2] / 255.0

    # Stack all channels
    combined = np.stack([
        rgb_norm[..., 0],  # R
        rgb_norm[..., 1],  # G
        rgb_norm[..., 2],  # B
        h_sin,  # H_sin
        h_cos,  # H_cos
        s_norm,  # S
        v_norm  # V
    ], axis=-1)

    return torch.from_numpy(combined.transpose(2, 0, 1))
def processor_with_6ch(filepath):
    img = processor(filepath)  # RGB image, already resized and cleaned
    if img is None:
        #print(f"Failed to load image: {filepath}")
        return None
    #converted = concat_6channel(img)
    converted = concat_6channel_model_fusion(img, input_range=(0, 255))
    #converted = convert_to_6ch_from_rgb(img)
    #weights = [0.5,2.0,0.5,1.0,0.5,0.5]  # RGB and HSV
    #converted = concat_6channel_WEIGHTS_with_CLAHE_YCBCR(img, input_range=(0, 255), weights=weights)


    #print(f"\nFile: {filepath}")
    #print(f"Shape of converted image: {converted.shape}")  # should be (6, H, W)

    #for i in range(6):
    #    print(
    #        f"Channel {i}: min={converted[i].min():.4f}, "
    #        f"max={converted[i].max():.4f}, "
    #        f"mean={converted[i].mean():.4f}, "
    #        f"nonzero={np.count_nonzero(converted[i])}"
    #    )

    return converted


PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed/psixchannel_probabilistic')
PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
PRO_TEST_DIR = PROCESS_DIR / 'test_images'

PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)



#this is a method that i use when preprocessing gets stuck in the server and i need to preprocess again the images
# that havent been preprocessed until a specific time cutoff_hour cutoff_minute
def get_images_needing_reprocessing(train_dir: Path, pro_train_dir: Path, cutoff_hour=22, cutoff_minute=0):
    cutoff_time = datetime.combine(date.today(), time(cutoff_hour, cutoff_minute))
    images_to_reprocess = []

    for npy_file in pro_train_dir.glob('*.npy'):
        modified_time = datetime.fromtimestamp(npy_file.stat().st_mtime)

        if modified_time < cutoff_time:
            png_file = train_dir / (npy_file.stem + '.png')
            if png_file.exists():
                images_to_reprocess.append(png_file)

    print(f"{len(images_to_reprocess)} images need reprocessing.")
    return images_to_reprocess


# Usage example:
#train_imgs = get_images_needing_reprocessing(TRAIN_DIR, PRO_TRAIN_DIR, cutoff_hour=11, cutoff_minute=0)


with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(processor_with_6ch, str(f)): f for f in train_imgs}
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
    results = {executor.submit(processor_with_6ch, str(f)): f for f in test_imgs}
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


def calculate_stats(image_dir):
    means = torch.zeros(7)
    stds = torch.zeros(7)
    image_files = list(image_dir.glob('*.npy'))

    for img_path in tqdm(image_files, desc='Calculating stats'):
        img = np.load(img_path)
        img_tensor = torch.from_numpy(img)
        means += img_tensor.mean(dim=(1, 2))  # Mean per channel
        stds += img_tensor.std(dim=(1, 2))  # Std per channel

    means /= len(image_files)
    stds /= len(image_files)

    print(f"Means: {means.tolist()}")
    print(f"Stds: {stds.tolist()}")
    return means, stds


# Calculate for training set
train_means, train_stds = calculate_stats(PRO_TRAIN_DIR)
# Save these values to use in your transforms
np.save(PROCESS_DIR / 'train_means.npy', train_means.numpy())
np.save(PROCESS_DIR / 'train_stds.npy', train_stds.numpy())


def load(filename):
    x = np.load(filename)
    return torch.from_numpy(x)



#means = np.zeros((6, len(pro_train_imgs)))
#mean_residuals = np.zeros((6, len(pro_train_imgs)))

'''
with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        img = future.result()
        f = results[future]
        index_map = {f: i for i, f in enumerate(pro_train_imgs)}
        idx = index_map[f]

        #idx = pro_train_imgs.index(f)

        # extract mean and variance for nonzero parts of image
        nonzero = (img > 0)
        for c in range(6):
            values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
            means[c, idx] = values.mean()

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        img = future.result()
        f = results[future]
        index_map = {f: i for i, f in enumerate(pro_train_imgs)}
        idx = index_map[f]

        #idx = pro_train_imgs.index(f)

        # extract mean and variance for nonzero parts of image
        nonzero = (img > 0)
        for c in range(6):
            values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
            if values.numel() > 0:
                mean_val = values.mean().item()
                means[c, idx] = mean_val
                mean_residuals[c, idx] = ((values - mean_val) ** 2).mean().item()
            else:
                means[c, idx] = 0.0
                mean_residuals[c, idx] = 0.0

#            mean_residuals[c, idx] = ((values - means[c, :].mean()) ** 2).mean()

for i in range(6):
    print(f"Channel {i} mean: {means[i, :].mean():.4f}")
for i in range(6):
    print(f"Channel {i} std: {np.sqrt(mean_residuals[i, :].mean()):.4f}")
'''
