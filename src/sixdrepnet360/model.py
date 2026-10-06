# SPDX-FileCopyrightText: 2023 Thorsten Hempel
# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ
#
# SPDX-License-Identifier: MIT

import math
from typing import cast

import torch
from torch import nn
from torchvision.models import resnet
from typing_extensions import override

from sixdrepnet360 import utils


class SixDRepNet360(nn.Module):
    inplanes: int
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    relu: nn.ReLU
    maxpool: nn.MaxPool2d
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    avgpool: nn.AvgPool2d
    linear_reg: nn.Linear

    def __init__(
        self, block: type[resnet.Bottleneck], layers: list[int], fc_layers: int = 1
    ) -> None:
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512 * block.expansion, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block: type[resnet.Bottleneck], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _i in range(1, blocks))

        return nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Module.__call__ is typed as returning Any, so each layer output is cast.
        x = cast(torch.Tensor, self.conv1(x))
        x = cast(torch.Tensor, self.bn1(x))
        x = cast(torch.Tensor, self.relu(x))
        x = cast(torch.Tensor, self.maxpool(x))

        x = cast(torch.Tensor, self.layer1(x))
        x = cast(torch.Tensor, self.layer2(x))
        x = cast(torch.Tensor, self.layer3(x))
        x = cast(torch.Tensor, self.layer4(x))

        x = cast(torch.Tensor, self.avgpool(x))
        x = x.view(x.size(0), -1)

        x = cast(torch.Tensor, self.linear_reg(x))
        return utils.compute_rotation_matrix_from_ortho6d(x)
