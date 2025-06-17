# lightweight_encoders.py
import torch
import torch.nn as nn

class SimpleThermalEncoder(nn.Module):
    def __init__(self, output_dim=256):
        """
        A lightweight CNN to encode a single-channel thermal image into a feature vector.
        Input: (batch_size, 1, 320, 320)
        Output: (batch_size, output_dim)
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 160, 160)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 80, 80)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 40, 40)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)), # -> (64, 1, 1)
            nn.Flatten() # -> (64)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)