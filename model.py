"""
This will define the CNN
"""
import torch
from torch import nn

"https://www.datacamp.com/tutorial/pytorch-cnn-tutorial"

# conv1 + ReLU -> conv2 + ReLU -> conv3 + ReLU

class CNN(nn.Module):
    def __init__(self) -> None:

        super().__init__()

        self.padding = nn.ConstantPad1d((3, 4), 0)

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=120, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=120, out_channels=120, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=120, out_channels=120, kernel_size=8)

        self.relu = nn.ReLU()

        self.fcl = nn.Linear(6000, 40)

        self.dropout = nn.Dropout(0.2)

        self.ll = nn.Linear(40, 1)

        self.fc_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        # Layer 1: conv1 + ReLU
        output_vals = self.padding(batch)
        output_vals = self.conv1(output_vals)
        output_vals = self.relu(output_vals)

        # Layer 2: conv2 + ReLU
        output_vals = self.padding(output_vals)
        output_vals = self.conv2(output_vals)
        output_vals = self.relu(output_vals)

        # Layer 3: conv3 + ReLU
        output_vals = self.padding(output_vals)
        output_vals = self.conv3(output_vals)
        output_vals = self.relu(output_vals)

        # Flatten:
        output_vals = torch.flatten(output_vals, start_dim=1)

        # Fully connected layer:
        output_vals = self.fcl(output_vals)
        output_vals = self.fc_relu(output_vals)

        # Dropout:
        output_vals = self.dropout(output_vals)

        # Linear layer:
        output_vals = self.ll(output_vals)

        return output_vals
