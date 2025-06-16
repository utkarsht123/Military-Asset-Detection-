import sys
import torch
from mobile_vit import create_lightweight_vit

if __name__ == "__main__":
    model = create_lightweight_vit('mobilevit_s', num_classes=8)
    print(model)
    print("\n--- Model Children ---")
    for name, module in model.named_children():
        print(f"{name}: {module}")
    print("\n--- Model Named Modules ---")
    for name, module in model.named_modules():
        print(name)
