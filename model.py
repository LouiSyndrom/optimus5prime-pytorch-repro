"""
This will define the CNN
"""
import torch
from torch import nn

"https://www.datacamp.com/tutorial/pytorch-cnn-tutorial"

class CNN(nn.Module):
    def __init__(self) -> None:

        super().__init__()

        self.padding = nn.ConstantPad1d((3, 4), 0)

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=120, kernel_size=8)

        self.relu = nn.ReLU()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        output_vals = self.padding(batch)

        output_vals = self.conv1(batch)

        output_vals = self.relu(output_vals)

        return output_vals