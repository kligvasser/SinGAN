import torch
import math
import torch.nn as nn
from utils.core import imresize
from copy import deepcopy
from torch.nn import functional as F

__all__ = ['g_multivanilla']

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
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.batch_norm(self.conv(x)))
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(Vanilla, self).__init__()
        # parameters
        self.padding = (kernel_size // 2) * num_blocks

        # features
        blocks = [BasicBlock(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding)]
        for i in range(0, num_blocks - 2):
            f = max_features // pow(2, (i+1))
            blocks.append(BasicBlock(in_channels=max(min_features, f * 2), out_channels=max(min_features, f), kernel_size=kernel_size, padding=padding))
        self.features = nn.Sequential(*blocks)
        
        # classifier
        self.features_to_image = nn.Sequential(
            nn.Conv2d(in_channels=max(f, min_features), out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.Tanh())
        
        # initialize weights
        initialize_model(self)

    def forward(self, z, x):
        z = F.pad(z, [self.padding, self.padding, self.padding, self.padding])
        z = self.features(z)
        z = self.features_to_image(z)
        
        return x + z

class MultiVanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(MultiVanilla, self).__init__()
        # parameters
        self.in_channels = in_channels
        self.max_features = max_features
        self.min_features = min_features
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.scale = 0
        self.key = 's0'
        self.scale_factor = 0

        # current
        self.curr = Vanilla(in_channels, max_features, min_features, num_blocks, kernel_size, padding)
        self.prev = nn.Module()

    def add_scale(self, device):
        self.scale += 1

        # previous
        self.prev.add_module(self.key, deepcopy(self.curr))
        self._reset_grad(self.prev, False)
        self.key = 's{}'.format(self.scale)

        # current
        max_features = min(self.max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(self.min_features * pow(2, math.floor(self.scale / 4)), 128)
        if math.floor(self.scale / 4) != math.floor((self.scale - 1) / 4):
            self.curr = Vanilla(self.in_channels, max_features, min_features, self.num_blocks, self.kernel_size, self.padding).to(device)

    def _compute_previous(self, reals, amps, noises=None):
        # parameters
        keys = list(reals.keys())
        y = torch.zeros_like(reals[keys[0]])
            
        # loop over scales
        for key, single_scale in self.prev.named_children():
            next_key = keys[keys.index(key) + 1]
            # fixed z
            if noises:
                z = y + amps[key].view(-1, 1, 1, 1) * noises[key]
            # random noise
            else:
                n = self._generate_noise(reals[key], repeat=(key == 's0'))
                z = y + amps[key].view(-1, 1, 1, 1) * n
            y = single_scale(z, y)
            y = imresize(y, 1. / self.scale_factor)
            y = y[:, :, 0:reals[next_key].size(2), 0:reals[next_key].size(3)]
            
        return y

    def forward(self, reals, amps, noises=None):
        # compute prevous layers
        with torch.no_grad():
            y = self._compute_previous(reals, amps, noises).detach()
            
        # fixed noise
        if noises:
            z = y + amps[self.key].view(-1, 1, 1, 1) * noises[self.key]
        # random noise
        else:
            n = self._generate_noise(reals[self.key], repeat=(not self.scale))
            z = y + amps[self.key].view(-1, 1, 1, 1) * n

        o = self.curr(z.detach(), y.detach()) 
        return o

    def _generate_noise(self, tensor_like, repeat=False):
        if not repeat:
            noise = torch.randn(tensor_like.size()).to(tensor_like.device)
        else:
            noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
            noise = noise.repeat((1, 3, 1, 1)).to(tensor_like.device)

        return noise

    def _reset_grad(self, model, require_grad=False):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def train(self, mode=True):
        self.training = mode
        # train
        for module in self.curr.children():
            module.train(mode)
        # eval
        for module in self.prev.children():
            module.train(False)
        return self

    def eval(self):
        self.train(False)

def g_multivanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    
    return MultiVanilla(**config)