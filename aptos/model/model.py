from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from aptos.base import BaseModel
import timm
from torch.nn.parameter import Parameter


def cycle_channel_layers(weights, n):
    """Repeat channel weights n times. Assumes channels are dim 1."""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights
'''
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
'''

#0.78
class EffNetOLD(BaseModel):
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
        #instead of x, have to do x_rgb, x_hsv
        # x=torch.cat x_rgb, x_hsv
        return self.base_model(x)


class EffNetHSV(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b2', verbose=0):
        super().__init__(verbose)

        # Load base model
        self.base_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        original_conv = self.base_model._conv_stem

        # Create two separate stems for RGB and HSV
        self.rgb_stem = nn.Conv2d(3, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        self.hsv_stem = nn.Conv2d(3, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        # Initialize weights
        with torch.no_grad():
            self.rgb_stem.weight[:] = original_conv.weight.clone()
            mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
            self.hsv_stem.weight[:] = mean_weight * 0.5

        # Replace stem with identity to skip it (we're handling stem manually)
        self.base_model._conv_stem = nn.Identity()

    def forward(self, x):
        # Split channels
        x_rgb = x[:, :3]
        x_hsv = x[:, 3:]

        # Apply custom dual stems
        rgb_out = self.rgb_stem(x_rgb)
        hsv_out = self.hsv_stem(x_hsv)

        x_combined = rgb_out + hsv_out

        # Inject into EfficientNet from next layer
        x = self.base_model._bn0(x_combined)
        x = self.base_model._swish(x)

        # Follow remaining blocks
        for block in self.base_model._blocks:
            x = block(x)
        x = self.base_model._conv_head(x)
        x = self.base_model._bn1(x)
        x = self.base_model._swish(x)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.base_model._dropout(x)
        x = self.base_model._fc(x)

        return x

#YCBCR
class EffNet(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b2', verbose=0):
        super().__init__(verbose)

        # Load base EfficientNet
        self.base_model = EfficientNet.from_name(f'efficientnet-{model}')
        model._fc = nn.Linear(model._fc.in_features, num_classes)  # Reset final layer

        #self.base_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        original_conv = self.base_model._conv_stem

        # Create two separate stems for RGB and YCbCr
        self.rgb_stem = nn.Conv2d(3, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        self.ycbcr_stem = nn.Conv2d(3, original_conv.out_channels,
                                    kernel_size=original_conv.kernel_size,
                                    stride=original_conv.stride,
                                    padding=original_conv.padding,
                                    bias=False)

        # Initialize RGB stem with original conv weights
        with torch.no_grad():
            self.rgb_stem.weight[:] = original_conv.weight.clone()

            # You could customize how much weight is given to Y, Cb, Cr
            mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
            self.ycbcr_stem.weight[:] = mean_weight * 0.5  # neutral init

        # Skip original stem
        self.base_model._conv_stem = nn.Identity()

    def forward(self, x):
        # Split input into RGB and YCbCr (first 3 + next 3 channels)
        x_rgb = x[:, :3]
        x_ycbcr = x[:, 3:]

        # Apply dual stems
        rgb_out = self.rgb_stem(x_rgb)
        ycbcr_out = self.ycbcr_stem(x_ycbcr)

        # Combine features
        x_combined = rgb_out + ycbcr_out

        # Continue through EfficientNet layers
        x = self.base_model._bn0(x_combined)
        x = self.base_model._swish(x)

        for block in self.base_model._blocks:
            x = block(x)

        x = self.base_model._conv_head(x)
        x = self.base_model._bn1(x)
        x = self.base_model._swish(x)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.base_model._dropout(x)
        x = self.base_model._fc(x)

        return x


class EffNetDual(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b2', verbose=0):
        super().__init__(verbose)

        # Load two EfficientNets
        self.rgb_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        self.hsv_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        self.rgb_model.set_swish(memory_efficient=False)
        self.hsv_model.set_swish(memory_efficient=False)

        # Modify HSV input
        original_conv = self.hsv_model._conv_stem
        self.hsv_model._conv_stem = nn.Conv2d(
            3,  # Now takes [H_sin, H_cos, S]
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Learnable fusion
        self.fusion_weight = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, x):
        # RGB path (channels 0-2)
        x_rgb = x[:, :3]

        # HSV path (channels 3=H_sin, 4=H_cos, 5=S)
        x_hsv = x[:, 3:6]  # Using precomputed features

        # Get outputs
        rgb_out = self.rgb_model(x_rgb)
        hsv_out = self.hsv_model(x_hsv)

        # Fusion
        w = torch.softmax(self.fusion_weight, dim=0)
        return w[0] * rgb_out + w[1] * hsv_out


#failed
class EffNetNEW(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b5', verbose=0):
        super().__init__(verbose)
        self.base_model = EfficientNet.from_pretrained(f'efficientnet-{model}', num_classes=num_classes)
        original_conv = self.base_model._conv_stem

        # Dual stems with better initialization
        self.rgb_stem = nn.Conv2d(3, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        self.hsv_stem = nn.Conv2d(3, original_conv.out_channels,
                                  kernel_size=original_conv.kernel_size,
                                  stride=original_conv.stride,
                                  padding=original_conv.padding,
                                  bias=False)

        # Improved initialization
        with torch.no_grad():
            self.rgb_stem.weight[:] = original_conv.weight.clone()
            hsv_weights = original_conv.weight[:, [0, 1, 2]] * torch.tensor([0.4, 0.4, 0.2])
            self.hsv_stem.weight[:] = hsv_weights

        # Learnable fusion
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * original_conv.out_channels, original_conv.out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(original_conv.out_channels // 8, 2, 1),
            nn.Softmax(dim=1)
        )

        self.base_model._conv_stem = nn.Identity()

    def forward(self, x):
        rgb = self.rgb_stem(x[:, :3])
        hsv = self.hsv_stem(x[:, 3:])

        combined = torch.cat([rgb, hsv], dim=1)
        weights = self.fusion(combined)  # shape: [B, 2, 1, 1]

        # Fix: expand fusion weights so they're broadcastable over [B, C, H, W]
        fused = rgb * weights[:, 0].unsqueeze(1) + hsv * weights[:, 1].unsqueeze(1)

        # Pass through the remaining EfficientNet (which includes features + classifier)
        x = self.base_model._bn0(fused)
        x = self.base_model._swish(x)

        for block in self.base_model._blocks:
            x = block(x)
        x = self.base_model._conv_head(x)
        x = self.base_model._bn1(x)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.base_model._dropout(x)
        x = self.base_model._fc(x)  # Final prediction (e.g., 1 value or 5-class softmax)
        return x


'''
class EffNet(BaseModel):
    """
    EfficientNet model adapted for 6-channel input.
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, num_classes, pretrained=True, model='b5', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'

        # Load the pretrained model
        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes) if pretrained \
                     else EfficientNet.from_name(model_name)

        ### 🔽 Modify input layer to accept 6 channels 🔽 ###
        # Save original conv weights
        w = self.model._conv_stem.weight  # shape [32, 3, 3, 3]

        # Define new Conv2d with 6 input channels
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=w.shape[0],
            kernel_size=w.shape[2:],
            stride=self.model._conv_stem.stride,
            padding=self.model._conv_stem.padding,
            bias=self.model._conv_stem.bias is not None
        )
        #  self.model._avg_pooling = GeM()

        with torch.no_grad():
            # Copy pretrained weights for the first 3 channels
            new_conv.weight[:, :3] = w
            # For the extra 3 channels: use the average of the original 3 channels
            new_conv.weight[:, 3:] = w.mean(dim=1, keepdim=True)

        # Replace original conv layer
        self.model._conv_stem = new_conv
        ### 🔼 Done modifying input layer 🔼 ###

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()
        self.logger.info(f'<init>: \n{self}')


'''
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

        self.logger.info(f'<init>: \n{self}')

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


class EffNetVit(BaseModel):
    def __init__(self, model='efficientvit_mit_b2', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model = 'efficientvit_b2.r256_in1k'
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        else:
            self.model = timm.create_model(model, pretrained=False, num_classes=num_classes)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class PVTv2(BaseModel):
    def __init__(self, model='pvt_v2_b2', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model = 'pvt_v2_b2'
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        else:
            self.model = timm.create_model(model, pretrained=False, num_classes=num_classes)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class NextViT(BaseModel):
    def __init__(self, model='nextvit_base', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model='nextvit_base'
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        else:
            self.model = timm.create_model(model, pretrained=False, num_classes=num_classes)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()
