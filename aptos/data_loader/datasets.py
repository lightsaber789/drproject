from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.distributions.beta import Beta
from torch.utils.data import Dataset
import torch.nn.functional as F


class PngDataset(Dataset):

    train_csv = 'train.csv'
    test_csv  = 'test.csv'

    def __init__(self, data_dir, transform, train=True):
        self.train = train
        self.transform = transform

        if self.train:
            self.images_dir = Path(data_dir) / 'train_images'
            self.labels_filename = Path(data_dir) / self.train_csv
        else:
            self.images_dir = Path(data_dir) / 'test_images'
            self.labels_filename = Path(data_dir) / self.test_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.png')

    @property
    def df(self):
        return self._df


    def load_img(self, filename):
        img = np.load(filename)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        return self.train_tsfm(img) if self.train else self.test_tsfm(img)

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        if not self.train:
            return x
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]


class NpyDataset(Dataset):

    train_csv = 'train.csv'
    test_csv  = 'test.csv'

    def __init__(self, data_dir, train_tsfm, test_tsfm, train=True, pseudo_dir=None):
        self.train = train
        self.train_tsfm = train_tsfm
        self.test_tsfm = test_tsfm

        if self.train:
            self.images_dir = Path(data_dir) / 'train_images'
            self.labels_filename = Path(data_dir) / self.train_csv
        else:
            self.images_dir = Path(data_dir) / 'test_images'
            self.labels_filename = Path(data_dir) / self.test_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.npy')

        # Load pseudo-labeled dataset if provided
        if pseudo_dir is not None and self.train:
            pseudo_labels_filename = Path(pseudo_dir)
            if pseudo_labels_filename.exists():
                pseudo_df = pd.read_csv(pseudo_labels_filename)
                pseudo_df['filename'] = pseudo_df['id_code'].apply(lambda x: data_dir / 'test_images' / f'{x}.npy')

                # Merge training and pseudo-labeled datasets
                self._df = pd.concat([self._df, pseudo_df], ignore_index=True)
                print(f"Dataset size after adding pseudo-labels: {len(self._df)} samples")
            else:
                print("Pseudo-labeled file not found, using only training data.")

    @property
    def df(self):
        return self._df

    #works
    def load_img_old(self, filename):
        '''
                img = torch.from_numpy(img).float()  # Convert to tensor first
                print("After permute:", img.shape)

                if img.ndim == 3 and img.shape[2] == 6:
                    img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                 Only permute if it's still a numpy array in (H, W, C)
                 '''
        img = np.load(filename)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 6:
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:
            img = torch.from_numpy(img).float()


        # Resize to (img_size, img_size)
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        assert img.shape[0] == 6, f"Expected 6 channels, got {img.shape[0]}"

        return self.train_tsfm(img) if self.train else self.test_tsfm(img)
    def load_imgoldold(self, filename):

        img = np.load(filename)
        img = torch.from_numpy(img).float()  # Convert to tensor first

        if img.ndim == 3 and img.shape[2] == 6:
            img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        # Resize to (img_size, img_size)
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

        return self.train_tsfm(img) if self.train else self.test_tsfm(img)

    def load_img(self, filename):
        img = np.load(filename)

        # Ensure the image is a float tensor
        img = torch.from_numpy(img).float()

        # If it's in (H, W, C) format with 7 channels, convert to (C, H, W)
        if img.ndim == 3 and img.shape[2] == 7:
            img = img.permute(2, 0, 1)

        # Assert after permute to catch wrong formats
        assert img.shape[0] == 7, f"Expected 7 channels, got {img.shape[0]}"

        # Apply transforms
        img = self.train_tsfm(img) if self.train else self.test_tsfm(img)

        return img

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]


class MixupNpyDataset(Dataset):

    N_CLASSES = 5
    train_csv = 'train.csv'

    def __init__(self, data_dir, transform, exclude_idx, alpha=0.4, train=True):
        self.images_dir = Path(data_dir) / 'train_images'
        self.labels_filename = Path(data_dir) / self.train_csv

        self.transform = transform
        self.train = train

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.npy')

        self.class_idxs = [
            [x for x in self.df.loc[self.df['diagnosis'] == c, :].index.values
                if x not in exclude_idx] for c in range(self.N_CLASSES)
        ]

        self.beta_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    @property
    def df(self):
        return self._df

    def random_float(self):
        return torch.rand((1,))[0].item()

    def random_choice(self, items):
        return items[torch.randint(len(items), (1,))[0].item()]

    def random_beta(self):
        """
        Return one-sided sample from beta distribution
        """
        sample = self.beta_dist.sample().item()
        if sample < 0.5:
            return sample
        sample -= 1
        return sample * -1

    def random_neighbour_class(self, c):
        if c == 0:
            return c if self.random_float() < 0.8 else c + 1
        if c == 4:
            return c if self.random_float() > 0.5 else c - 1
        return c - 1 if self.random_float() > 0.5 else c + 1

    def load_img(self, filename):
        img = np.load(filename)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        return self.train_tsfm(img) if self.train else self.test_tsfm(img)

    def mixup(self, X1, X2, y1, y2):
        alpha = self.random_beta()
        beta = 1 - alpha
        X = (alpha * X1) + (beta * X2)
        y = (alpha * y1) + (beta * y2)
        return X, y

    def __getitem__(self, idx1):
        y1 = self.df.iloc[idx1]['diagnosis']

        if self.train:
            y2 = self.random_neighbour_class(y1)
            idx2 = self.random_choice(self.class_idxs[y2])
            assert y2 == self.df.iloc[idx2]['diagnosis']
        else:  # mixup with self
            y2 = y1
            idx2 = idx1

        f1 = self.df.iloc[idx1]['filename']
        f2 = self.df.iloc[idx2]['filename']

        X1 = self.load_img(f1)
        X2 = self.load_img(f2)

        X, y = self.mixup(X1, X2, y1, y2)
        return (X, y)

    def __len__(self):
        return self.df.shape[0]
