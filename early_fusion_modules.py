# early_fusion_modules.py
import torch
import torch.nn as nn

class ConcatenationFusion(nn.Module):
    def __init__(self, rgb_feature_dim, thermal_feature_dim, hidden_dim=512, output_dim=256):
        """
        A simple fusion module that concatenates two feature vectors and
        passes them through a small MLP.
        """
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(rgb_feature_dim + thermal_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, rgb_features, thermal_features):
        # Concatenate along the feature dimension
        fused_features = torch.cat((rgb_features, thermal_features), dim=1)
        return self.fusion_layer(fused_features)