from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import torchvision.transforms as T
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop(img):
    """
    Create circular crop around image centre
    """
    img = crop_image_from_gray(img)
    height, width, depth = img.shape
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img

# Paths
DATA_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/raw')
TRAIN_DIR = DATA_DIR / 'train_images'
TEST_DIR = DATA_DIR / 'test_images'
NWORKERS = 8

train_imgs = list(TRAIN_DIR.glob('*.png'))
test_imgs = list(TEST_DIR.glob('*.png'))
print(len(train_imgs), len(test_imgs))


PROCESS_DIR = Path('/src/rfir/project/aptos2019-blindness-detection/data/preprocessed/kaggle')
PRO_TRAIN_DIR = PROCESS_DIR / 'train_images'
PRO_TEST_DIR = PROCESS_DIR / 'test_images'

PRO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PRO_TEST_DIR.mkdir(parents=True, exist_ok=True)

def processor(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")  # Log skipped images
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = circle_crop(img)
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None




# Process and save training images
with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(processor, str(f)): f for f in train_imgs}
    for future in tqdm(as_completed(results), total=len(train_imgs)):
        try:
            img = future.result()
            if img is None:
                continue
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
                continue
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

# made server lag
with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        try:
            img = future.result()
            f = results[future]
            idx = pro_train_imgs.index(f)
            print(f'Processing image {idx + 1}/{len(pro_train_imgs)}')

            # Extract mean for nonzero parts of image
            nonzero = img > 0
            for c in range(3):
                values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
                means[c, idx] = values.mean()
        except Exception as e:
            print(f'Error processing image {f}: {e}')

with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
    results = {executor.submit(load, str(f)): f for f in pro_train_imgs}
    for future in tqdm(as_completed(results), total=len(pro_train_imgs)):
        try:
            img = future.result()
            f = results[future]
            idx = pro_train_imgs.index(f)
            print(f'Processing image {idx + 1}/{len(pro_train_imgs)} for mean residuals')

            # Compute mean residuals
            nonzero = img > 0
            for c in range(3):
                values = img[c, :, :].view(-1)[nonzero[c, :, :].view(-1)]
                mean_residuals[c, idx] = ((values - means[c, :].mean()) ** 2).mean()
        except Exception as e:
            print(f'Error processing image {f}: {e}')

print(means[0, :].mean())
print(means[1, :].mean())
print(means[2, :].mean())
print(np.sqrt(mean_residuals[0, :].mean()))
print(np.sqrt(mean_residuals[1, :].mean()))
print(np.sqrt(mean_residuals[2, :].mean()))

