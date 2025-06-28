import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB4,EfficientNetB5
from tensorflow.keras import backend as K

#model = EfficientNetB3(weights='imagenet')
import json
import math
from tqdm import tqdm #, tqdm_notebook
import gc
import warnings
import os
import cv2
from PIL import Image
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from skimage.color import rgb2hsv, lab2lch
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
## io-related
from skimage.io import imread
import os
from glob import glob

#import tensorflow_addons as tfa
from keras.layers import Dense, Dropout

import torch
import timm as timm
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import torchvision.transforms as T

from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler

import sys
sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
from efficientnet_pytorch import EfficientNet

print("Done")


class DataLoaderBase(DataLoader):

    def __init__(self, dataset, batch_size, epoch_size, validation_split, num_workers,
                 train=True, alpha=None, verbose=0):
        self.verbose = verbose
        self.ids = dataset.df['id_code'].values

        self.sampler, self.valid_sampler = self._setup_samplers(
            dataset,
            batch_size,
            epoch_size,
            validation_split,
            alpha)

        init_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **init_kwargs)

    def _setup_samplers(self, dataset, batch_size, epoch_size, validation_split, alpha):
        # get sampler & indices to use for validation
        valid_sampler, valid_idx = self._setup_validation(dataset, batch_size, validation_split)

        # get sampler & indices to use for training/testing
        train_sampler, n_samples = self._setup_train(
            dataset, batch_size, epoch_size, alpha, valid_idx)
        self.n_samples = n_samples

        return (train_sampler, valid_sampler)

    def _setup_train(self, dataset, batch_size, epoch_size, alpha, exclude_idx):
        all_idx = np.arange(len(dataset))
        train_idx = [i for i in all_idx if i not in exclude_idx]

        if alpha is None:
            subset = Subset(dataset, train_idx)
            sampler = BatchSampler(SequentialSampler(subset), batch_size, False)
            return sampler, len(train_idx)

        factory = SamplerFactory(self.verbose)
        sampler = factory.get(dataset.df, train_idx, batch_size, epoch_size, alpha)
        return sampler, len(sampler) * batch_size

    def _setup_validation(self, dataset, batch_size, split):
        if split == 0.0:
            return None, []
        all_idx = np.arange(len(dataset))
        len_valid = int(len(all_idx) * split)
        valid_idx = np.random.choice(all_idx, size=len_valid, replace=False)
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size, False)
        valid_targets = dataset.df.iloc[valid_idx].groupby('diagnosis').count()
        return valid_sampler, valid_idx


class PngDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, alpha=None, verbose=0):
        transform = InplacePngTransforms(train, img_size)
        dataset = PngDataset(data_dir, transform, train=train)

        super().__init__(dataset, batch_size, None, validation_split, num_workers,
                         train=train, alpha=alpha, verbose=verbose)


class AugmentationBase:

    def __init__(self, train):
        self.train = train
        self.transform = self.build_transforms()

    def build_transforms(self):
        raise NotImplementedError('Not implemented!')

    def __call__(self, images):
        return self.transform(images)


class MediumNpyTransforms(AugmentationBase):
    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(self.img_size, scale=(0.9, 0.98)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
        ])


class InplacePngTransforms(AugmentationBase):

    def __init__(self, train, img_size):
        self.img_size = img_size
        self.processor = ImgProcessor()
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            self.processor,
            MediumNpyTransforms(self.train, self.img_size)
        ])


print("done")


class ImgProcessor:
    """
    This class is responsible for preprocessing the images, eg. crop, sharpen, resize, normalise.
    """

    def __init__(self, crop_tol=12, img_width=600, verbose=0):
        self.crop_tol = crop_tol
        self.img_width = img_width
        self.sequential = T.Compose([
            self.read_png,
            self.crop_black,
            self.crop_square,
            self.resize
        ])

    def __call__(self, filename):
        return self.sequential(filename)

    def read_png(self, filename):
        """
        Load the image into a numpy array, and switch the channel order so it's in the format
        expected by matplotlib (rgb).
        """
        return cv2.imread(filename)[:, :, ::-1]  # bgr => rgb

    def crop_black(self, img):
        """
        Apply a bounding box to crop empty space around the image. In order to find the bounding
        box, we blur the image and then apply a threshold. The blurring helps avoid the case where
        an outlier bright pixel causes the bounding box to be larger than it needs to be.
        """
        gb = cv2.GaussianBlur(img, (7, 7), 0)
        mask = (gb > self.crop_tol).any(2)
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img[y0:y1, x0:x1]

    def crop_square(self, img):
        """
        Crop the image to a square (cutting off sides of a circular image).
        """
        H, W, C = img.shape
        crop_size = min(int(W * 0.87), H)
        if W <= crop_size:
            x0 = 0
            x1 = W
        else:
            width_excess = W - crop_size
            x0 = width_excess // 2
            x1 = min(x0 + crop_size, W)
        if H <= crop_size:
            y0 = 0
            y1 = H
        else:
            height_excess = H - crop_size
            y0 = height_excess // 2
            y1 = min(y0 + crop_size, H)
        return img[y0:y1, x0:x1]

    def resize(self, img):
        dim = (self.img_width, self.img_width)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


print("done")


class PngDataset(Dataset):
    train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    test_csv = "/kaggle/input/aptos2019-blindness-detection/test.csv"

    def __init__(self, data_dir, transform, train=False):
        self.train = train
        self.transform = transform

        if self.train:
            self.images_dir = Path("/kaggle/input/aptos2019-blindness-detection") / 'train_images'
            self.labels_filename = Path("/kaggle/input/aptos2019-blindness-detection") / self.train_csv
        else:
            self.images_dir = Path("/kaggle/input/aptos2019-blindness-detection") / 'test_images'
            self.labels_filename = Path("/kaggle/input/aptos2019-blindness-detection") / self.test_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.png')

    @property
    def df(self):
        return self._df

    def load_img(self, filename):
        try:
            return self.transform(str(filename))  # let transforms do loading
        except:
            return torch.zeros((3, 256, 256))

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        if not self.train:
            return x
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]


print("Done")
config_path = '/kaggle/input/config-yml-agian/config.yml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# def get_instance(name, config, *args):
#    return getattr(EffNet, config[name]['type'])(*args, **config[name]['args'])
def get_instance(cls, config, *args):
    return cls(*args, **config['args'])  # Directly instantiate the class


print(config)
print("-----------------------------")
print(config['arch']['type'])
print(config['arch']['args'])


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, verbose=0):
        super().__init__()

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'


def cycle_channel_layers(weights, n):
    """Repeat channel weights n times. Assumes channels are dim 1."""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights


class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'

        weight_path = "/kaggle/input/efficientnetb2_weights/pytorch/default/1/efficientnet-b2-8bb594d6.pth"

        # ✅ Load the model architecture
        self.model = EfficientNet.from_name(model_name, num_classes=1000)  # Use default num_classes first

        if pretrained and weight_path is not None:
            print(f"🔄 Loading weights from: {weight_path}")
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))

            # ✅ Ignore `_fc` weights to avoid shape mismatch
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('_fc')}
            self.model.load_state_dict(filtered_checkpoint, strict=False)
            print("✅ Pretrained weights loaded (excluding `_fc`).")

            # ✅ Modify `_fc` layer to match `num_classes`
            in_features = self.model._fc.in_features
            self.model._fc = torch.nn.Linear(in_features, num_classes)

        else:
            self.model = EfficientNet.from_name(
                model_name, override_params={'num_classes': num_classes}
            )
        # for name, w in self.model.named_parameters():
        #     if '_fc' not in name:
        #         w.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EffNetMaxAvg(BaseModel):

    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)

        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNetMaxAvg.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNetMaxAvg.from_name(
                model_name,
                override_params={'num_classes': num_classes})

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EfficientNetMaxAvg(EfficientNet):
    """
    Modified EfficientNet to use concatenated Max + Avg pooling
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.model._bn1.num_features * 2)),
            ('drop1', nn.Dropout(p=self._dropout)),
            ('linear1', nn.Linear(self.model._bn1.num_features * 2, 512)),
            ('mish', Mish()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=self._dropout / 2)),
            ('linear2', nn.Linear(512, self._global_params.num_classes))
        ]))

        nn.init.kaiming_normal_(fc._modules['linear1'].weight)
        nn.init.kaiming_normal_(fc._modules['linear2'].weight)

        self._bn1 = AdaptiveMaxAvgPool()
        self._fc = fc

    def forward(self, x):
        x = self.extract_features(x)
        x = self._bn1(x)
        x = self._fc(x)
        return x


class AdaptiveMaxAvgPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        return x


class Mish(nn.Module):
    """
    https://github.com/lessw2020/mish/blob/master/mish.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def get_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def prepare_device(model, n_gpu_use):
    device, device_ids = get_device(n_gpu_use)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, device


print("Done")

img_size = 256

data_loader = PngDataLoader(
    config['testing']['data_dir'],
    batch_size=config['testing']['batch_size'],
    validation_split=0.0,
    train=False,
    alpha=None,
    img_size=config['testing']['img_size'],
    num_workers=config['testing']['num_workers'],
    verbose=config['testing']['verbose']
)

#model = get_instance(EffNet,'arch', config)
model = get_instance(EffNet, config['arch'])
model, device = prepare_device(model, config['n_gpu'])
model_checkpoint = '../input/checkpointnnsmoothl1loss20epoch/pytorch/default/1/checkpoint-epoch20.pth'

checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
# prepare model for testing
model.eval()


pred_df = pd.DataFrame({'id_code': data_loader.ids})

with torch.no_grad():
    for e in range(2):  # perform N sets of predictions and average results
        preds = torch.zeros(len(data_loader.dataset))
        for i, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            output = output.detach().cpu()
            batch_size = output.shape[0]
            preds[i * batch_size:(i + 1) * batch_size] = output.squeeze(1)

        # add column for this iteration of predictions
        pred_df[str(e)] = preds.numpy()

# wrangle predictions
pred_df.set_index('id_code', inplace=True)
pred_df['diagnosis'] = pred_df.apply(lambda row: int(np.round(row.mean())), axis=1)

# pred_df.to_csv('preds.csv')
pred_df[['diagnosis']].to_csv('submission.csv')
