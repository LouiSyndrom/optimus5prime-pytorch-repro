# Adapted from feature/CNN_optimized by Yefeng Lin.

from pathlib import Path

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        return self.gelu(out + x)


class MultiscaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(4, 64, kernel_size=11, padding=5)
        self.merge = nn.Conv1d(192, 128, kernel_size=1)
        self.bn = nn.BatchNorm1d(128)
        self.gelu = nn.GELU()
        self.block1 = ResidualBlock(128, dilation=1)
        self.block2 = ResidualBlock(128, dilation=2)
        self.block3 = ResidualBlock(128, dilation=4)
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.2)
        self.mrl_head = nn.Linear(32, 1)
        self.fraction_head = nn.Linear(32, 14)

    def forward(self, x):
        short = self.gelu(self.conv3(x))
        medium = self.gelu(self.conv7(x))
        long = self.gelu(self.conv11(x))
        x = torch.cat((short, medium, long), dim=1)
        x = self.gelu(self.bn(self.merge(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.gelu(self.fc2(x))
        return self.mrl_head(x).squeeze(1), self.fraction_head(x)


def save_cnn_model(model, path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["model_state"] = model.state_dict()
    torch.save(checkpoint, path)


def load_cnn_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = MultiscaleCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
