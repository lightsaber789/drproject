import torchvision.transforms as T
import torch

from .preprocess import ImgProcessor

import torch.nn as nn

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
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

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


#perplexityai
class HeavyNpyTransforms(AugmentationBase):
    def __init__(self, train, img_size):
        self.img_size = img_size
        # Use your calculated statistics
        self.MEANS = [0.5304653644561768, 0.2818927764892578, 0.08917295187711716, 0.22496990859508514, 0.9450165629386902, 0.7896173000335693, 0.5307199954986572]  # Rounded values
        self.STDS = [0.1443122923374176, 0.08575006574392319, 0.040803492069244385, 0.0700300931930542, 0.1826087087392807, 0.12851780652999878, 0.1441347897052765]  # Rounded values
        super().__init__(train)

    def build_transforms(self):
        transforms = [
            # Tensor-compatible augmentations
            T.ToTensor(),  # Converts numpy array to tensor first
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            self.CustomColorJitter(brightness=0.2, contrast=0.2),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),

            # 7-channel normalization (MUST BE LAST)
            T.Normalize(mean=self.MEANS, std=self.STDS)
        ]
        return T.Compose(transforms)

class CustomColorJitter:
    """Only affects RGB (channels 0-2) and V (channel 6)"""

    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x):
        if self.brightness > 0:
            brightness_factor = torch.tensor(1.0).uniform_(
                1 - self.brightness, 1 + self.brightness)
            x[[0, 1, 2, 6]] *= brightness_factor

        if self.contrast > 0:
            contrast_factor = torch.tensor(1.0).uniform_(
                1 - self.contrast, 1 + self.contrast)
            mean = x[:3].mean()
            x[:3] = (x[:3] - mean) * contrast_factor + mean
        return x


class TensorOnlyTransforms(nn.Module):
    def __init__(self, train=True):
        super().__init__()
        self.train = train

    def forward(self, x):
        if not self.train:
            return x
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()  # Convert numpy to tensor

        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])  # vertical flip
        return x

#old one that works
class HeavyNpyTransformsOld(AugmentationBase):

    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.RandomAffine(
            #     degrees=5,
            #     translate=(0.03, 0.0),
            #     # shear=(0.05),
            #     # fillcolor=(128, 128, 128)
            # ),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 0.95)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            T.RandomErasing(
                p=0.8,
                scale=(0.05, 0.15),
                ratio=(0.4, 2.5)
            )
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
class MixupNpyTransforms(AugmentationBase):

    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

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
            T.RandomErasing(
                p=0.20,
                scale=(0.03, 0.08),
                ratio=(0.5, 2.0)
            )
        ])
