# mobile_vit.py
import torch.nn as nn
import timm

def create_lightweight_vit(model_name, num_classes, pretrained=True):
    """
    Creates a MobileViT model with a custom classification head.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes # timm will replace the head for us
    )
    print(f"Loaded {model_name} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    return model