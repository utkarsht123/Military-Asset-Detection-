# lightweight_multimodal_detr.py
import torch
import torch.nn as nn
from mobilevit_backbone import create_mobilevit_backbone
from lightweight_encoders import SimpleThermalEncoder
from early_fusion_modules import ConcatenationFusion

class LightweightMultiModalDETR(nn.Module):
    def __init__(self, num_classes, num_queries=100):
        """
        Integrates multimodal backbones with a simple DETR detection head.
        """
        super().__init__()
        # 1. Backbones & Encoders
        self.rgb_backbone = create_mobilevit_backbone()
        rgb_feature_dim = self.rgb_backbone.feature_info.channels()[-1]
        thermal_feature_dim = 256
        
        self.thermal_encoder = SimpleThermalEncoder(output_dim=thermal_feature_dim)
        
        # 2. Fusion Module
        fused_dim = 512
        self.fusion_module = ConcatenationFusion(
            rgb_feature_dim=rgb_feature_dim,
            thermal_feature_dim=thermal_feature_dim,
            output_dim=fused_dim
        )
        
        # 3. Simple DETR-like Head
        self.query_embed = nn.Embedding(num_queries, fused_dim)
        self.transformer = nn.Transformer(d_model=fused_dim, nhead=8, 
                                          num_encoder_layers=2, num_decoder_layers=2,
                                          dim_feedforward=1024, dropout=0.1)
        
        # 4. Prediction Heads
        self.class_head = nn.Linear(fused_dim, num_classes + 1) # +1 for "no object" class
        self.bbox_head = nn.Linear(fused_dim, 4) # (cx, cy, w, h)

        # 5. Helper layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.input_proj = nn.Conv2d(rgb_feature_dim, fused_dim, kernel_size=1) # Project to fused_dim

    def forward(self, rgb_image, thermal_image):
        # Process RGB image
        rgb_features_map = self.rgb_backbone(rgb_image)[-1] # Use last feature map

        # Project features to match transformer dimension
        proj_features = self.input_proj(rgb_features_map)
        bs, c, h, w = proj_features.shape
        
        # Pass through Transformer Encoder
        # Reshape for transformer: (seq_len, batch, dim)
        encoded_memory = self.transformer.encoder(proj_features.flatten(2).permute(2, 0, 1))
        
        # Process Thermal image (global features)
        thermal_features = self.thermal_encoder(thermal_image)
        
        # Fuse with global thermal features (simple addition to all tokens)
        # This is a basic way to inject global context.
        fused_memory = encoded_memory #+ self.fusion_module.fusion_layer[0](thermal_features).unsqueeze(0)

        # Pass through Transformer Decoder
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer.decoder(tgt, fused_memory, memory_key_padding_mask=None)
        
        # Get Predictions
        # hs shape: (num_queries, batch_size, embed_dim)
        hs = hs.permute(1, 0, 2)  # (batch_size, num_queries, embed_dim)
        pred_logits = self.class_head(hs)  # (batch_size, num_queries, num_classes+1)
        pred_boxes = self.bbox_head(hs).sigmoid()  # (batch_size, num_queries, 4)

        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}