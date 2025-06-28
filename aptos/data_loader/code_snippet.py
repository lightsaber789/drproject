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


#this function is used to preprocess images and save them into a folder
def convert_to_6ch_from_rgb(img):
    # Get channels in float32
    if img.shape[0] == 3:  # (C,H,W)
        R, G, B = img[0].astype(np.float32), img[1].astype(np.float32), img[2].astype(np.float32)
    else:  # (H,W,C)
        R, G, B = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)

    # 1. Medical-optimized RGB combinations (changed ratios)
    RG = (0.85 * R + 0.15 * G).clip(0, 255)  # More emphasis on red (hemorrhages)
    RB = (0.95 * R + 0.05 * B).clip(0, 255)  # Strong red focus (blood vessels)
    GB = (0.1 * G + 0.9 * B).clip(0, 255)  # Blue emphasis (exudates)

    # 2. HSV conversion with CLAHE enhancement (NON-NEGOTIABLE)
    rgb_uint8 = np.stack([R.clip(0, 255), G.clip(0, 255), B.clip(0, 255)], axis=-1).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]  # Note: V is uint8 here

    # --- CLAHE CRITICAL SECTION ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 8x8 grid works best for fundus
    V_enhanced = clahe.apply(V).astype(np.float32)  # Enhanced V channel
    # -----------------------------

    # 3. HSV-based combinations using ENHANCED V channel
    HS = ((H / 180.0) * 0.7 + (S / 255.0) * 0.3).clip(0, 1)  # Hue-Saturation (unchanged)
    SV = ((S / 255.0) * 0.3 + (V_enhanced / 255.0) * 0.7).clip(0, 1)  # More weight to enhanced V
    HV = ((H / 180.0) * 0.4 + (V_enhanced / 255.0) * 0.6).clip(0, 1)  # Balanced Hue/Value

    return np.stack([
        RG / 255.0,  # Channel 0
        RB / 255.0,  # Channel 1
        GB / 255.0,  # Channel 2
        HS,  # Channel 3 (unchanged)
        SV,  # Channel 4 (now with enhanced V)
        HV  # Channel 5 (now with enhanced V)
    ], axis=0).astype(np.float32)

# this is the model
class EffNet(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b2', verbose=0):
        super().__init__(verbose)

        # Load model
        self.base_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        original_conv = self.base_model._conv_stem

        # Learnable channel weights
        self.rg_weight = nn.Parameter(torch.tensor([0.7, 0.3]))
        self.rb_weight = nn.Parameter(torch.tensor([0.8, 0.2]))
        self.gb_weight = nn.Parameter(torch.tensor([0.2, 0.8]))

        # Modified stem
        self.new_stem = nn.Conv2d(6, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        # Initialize weights
        with torch.no_grad():
            # First 3 channels (learnable weighted combos)
            self.new_stem.weight[:, 0] = self.rg_weight[0] * original_conv.weight[:, 0] + \
                                         self.rg_weight[1] * original_conv.weight[:, 1]

            self.new_stem.weight[:, 1] = self.rb_weight[0] * original_conv.weight[:, 0] + \
                                         self.rb_weight[1] * original_conv.weight[:, 2]

            self.new_stem.weight[:, 2] = self.gb_weight[0] * original_conv.weight[:, 1] + \
                                         self.gb_weight[1] * original_conv.weight[:, 2]

            # Last 3 channels
            self.new_stem.weight[:, 3] = 0.5 * (original_conv.weight[:, 0] + original_conv.weight[:, 1])
            self.new_stem.weight[:, 4] = original_conv.weight.mean(dim=1) * 0.33
            self.new_stem.weight[:, 5] = original_conv.weight.mean(dim=1)

        self.base_model._conv_stem = self.new_stem

    def forward(self, x):
        return self.base_model(x)


class TimmEncoder(nn.Module):
def init(self, model_name, in_channels=3, depth=None):
super().init()
model = timm.create_model(model_name, features_only=True, in_chans=in_channels, pretrained=True)
depth = depth or len(model.feature_info)
self.model = timm.create_model(model_name, features_only=True, in_chans=in_channels, out_indices=list(range(depth)), pretrained=True)
self.out_channels = [f['num_chs'] for f in self.model.feature_info]

def forward(self, x):
    return self.model(x)

