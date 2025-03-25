import torch
import torch.nn as nn

class Smish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(out_channels),
                        Smish())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.activation = Smish()
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out

class LocalStage(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[1,1,1,1], output_dim=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
                        nn.BatchNorm2d(64),
                        Smish())
        self.layer0 = self._make_layer(block, 96, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 256, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 384, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[3], stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*3*256, out_features=1024),
            nn.BatchNorm1d(1024),
            Smish(),
            nn.Linear(in_features=1024, out_features=output_dim)
        )
    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, padding, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.layer0(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool2(x)
        x = self.fc(x)
        return x