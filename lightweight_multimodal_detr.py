# lightweight_multimodal_detr.py
import torch
import torch.nn as nn
from mobilevit_backbone import create_mobilevit_backbone
from lightweight_encoders import SimpleThermalEncoder
from early_fusion_modules import ConcatenationFusion

class LightweightMultiModalDETR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_backbone = create_mobilevit_backbone()
        
        # Define feature dimensions
        rgb_feature_dim = self.rgb_backbone.feature_info.channels()[-1] 
        thermal_feature_dim = 256
        fused_dim = 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.thermal_encoder = SimpleThermalEncoder(output_dim=thermal_feature_dim)
        self.fusion_module = ConcatenationFusion(
            rgb_feature_dim=rgb_feature_dim,
            thermal_feature_dim=thermal_feature_dim,
            output_dim=fused_dim
        )
        
        # --- Detection Heads ---
        # A simple MLP head for class prediction
        self.class_head = nn.Linear(fused_dim, num_classes + 1) # +1 for "no object" class
        # A simple MLP head for bounding box prediction [x_center, y_center, width, height]
        self.bbox_head = nn.Linear(fused_dim, 4)

    def forward(self, rgb_image, thermal_image):
        # Process RGB
        rgb_features_maps = self.rgb_backbone(rgb_image)
        rgb_features_last_stage = rgb_features_maps[-1]
        
        # To simplify, we will do detection on the whole image feature vector
        # A real DETR would use a transformer decoder on patch embeddings.
        # This is a "Lightweight" interpretation.
        rgb_pooled = self.pool(rgb_features_last_stage)
        rgb_flat = self.flatten(rgb_pooled)

        # Process Thermal
        thermal_features = self.thermal_encoder(thermal_image)

        # Fuse features
        fused_features = self.fusion_module(rgb_flat, thermal_features)
        
        # Make predictions
        class_logits = self.class_head(fused_features)
        pred_boxes = self.bbox_head(fused_features).sigmoid() # Sigmoid to keep outputs in [0, 1]
        
        return {'pred_logits': class_logits, 'pred_boxes': pred_boxes}