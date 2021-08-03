import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

__all__ = ['d_vanilla', 'd_snvanilla']

def initialize_model(model, scale=1.):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            continue

class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, normalization=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if normalization:
            self.conv = SpectralNorm(self.conv)

    def forward(self, x):
        x = self.lrelu(self.batch_norm(self.conv(x)))
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding, normalization):
        super(Vanilla, self).__init__()
        # features
        blocks = [BasicBlock(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding, normalization=normalization)]
        for i in range(0, num_blocks - 2):
            f = max_features // pow(2, (i+1))
            blocks.append(BasicBlock(in_channels=max(min_features, f * 2), out_channels=max(min_features, f), kernel_size=kernel_size, padding=padding, normalization=normalization))
        self.features = nn.Sequential(*blocks)
        
        # classifier
        self.classifier = nn.Conv2d(in_channels=max(f, min_features), out_channels=1, kernel_size=kernel_size, padding=padding)
        
        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def d_vanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    config.setdefault('normalization', False)
    
    return Vanilla(**config)

def d_snvanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    config.setdefault('normalization', True)
    
    return Vanilla(**config)