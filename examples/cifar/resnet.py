import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from candle import SerializableModule
import candle

class BasicBlock(SerializableModule):
    def __init__(self, in_planes, planes, prune_cfg, dropout=0, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_planes, momentum=0.9),
            nn.ELU(),
            candle.PruneConv2d((in_planes, planes, 3), prune_cfg, padding=1),
            nn.Dropout(dropout),
            nn.BatchNorm2d(planes, momentum=0.9),
            nn.ELU(),
            candle.PruneConv2d((planes, planes, 3), prune_cfg, padding=1, stride=stride)
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.skip = nn.Conv2d(in_planes, planes, 1, stride=stride)

    def forward(self, x):
        return self.skip(x) + self.layer(x)

class WideResNet(SerializableModule):
    def __init__(self):
        super().__init__()
        def make_layer(layer_idx, stride=1):
            in_planes = 16 if layer_idx == 0 else 2**(layer_idx - 1) * 16 * k
            planes = 2**layer_idx * 16 * k
            blocks = [BasicBlock(planes, planes, prune_cfg, dropout=0.3) for _ in range(n)]
            blocks[0] = BasicBlock(in_planes, planes, prune_cfg, dropout=0.3, stride=stride)
            return nn.Sequential(*blocks)
        prune_cfg = candle.read_config()
        k = 10
        n = 7
        self.conv1 = candle.PruneConv2d((3, 16, 3), prune_cfg, padding=1)
        self.conv2 = make_layer(0)
        self.conv3 = make_layer(1, 2)
        self.conv4 = make_layer(2, 2)
        self.output = candle.PruneLinear((512, 100), prune_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        return self.output(x)