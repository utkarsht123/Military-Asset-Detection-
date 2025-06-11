import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def get_vit_model(num_labels: int = 2, pretrained: bool = True):
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_labels
        )
    else:
        config = ViTConfig(num_labels=num_labels)
        model = ViTForImageClassification(config)
    return model
