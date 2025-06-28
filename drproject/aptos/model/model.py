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
GeM did not improve my Kaggle score
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

#0.78  six channel rgb with hsv split into rg bg rb hs hv sv. training on preprocessed files with function convert_to_6ch_from_rgb in preprocess-6channel.py
class EffNetOLD(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b5', verbose=0):
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

# training on preprocessed images by function concat_6channel in preprocess-6channel
class EffNetHSV(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b5', verbose=0):
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

#YCBCR rgb concat. result 0.76. weights Green‐heavy bolsters blood‐vessel contrast (critical for microaneurysms and vessel abnormalities), with moderate Y for optic‐disc clarity. Vessel & Optic Disk Focus [0.5,2.0,0.5,1.0,0.5,0.5]
class EffNet(BaseModel):
    def __init__(self, num_classes, pretrained=True, model='b5', verbose=0):
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




'''
model for RGB scenario 0.88. see preprocessing file preprocess.py

class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(
                model_name,
                override_params={'num_classes': num_classes})

        # for name, w in self.model.named_parameters():
        #     if '_fc' not in name:
        #         w.requires_grad = False

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


'''
#never used
class EffNetMaxAvg(BaseModel):

    def __init__(self, num_classes, pretrained, model='b5', verbose=0):
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

#never used
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

#never used
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

#never used
class Mish(nn.Module):
    """
    https://github.com/lessw2020/mish/blob/master/mish.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


#trying to use transformers
#effnet transformer gave the same result with RGB images training as EfficientNetb2 with RGB images Private score: 0.874812
class EffNetVit(BaseModel):
    def __init__(self, model='efficientvit_mit_b5', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model = 'efficientvit_b5.r256_in1k'
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



#didnt check score on kaggle
class PVTv2(BaseModel):
    def __init__(self, model='pvt_v2_b5', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model = 'pvt_v2_b5'
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
