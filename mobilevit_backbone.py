# mobilevit_backbone.py
import timm
import torch.nn as nn

def create_mobilevit_backbone(model_name='mobilevit_s', pretrained=True):
    """
    Creates a MobileViT model with the classification head removed,
    so it can be used as a feature extractor backbone.
    """
    # features_only=True returns feature maps from multiple stages
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
    )
    print(f"Created MobileViT backbone. Feature dimension: {backbone.feature_info.channels()}")
    return backbone