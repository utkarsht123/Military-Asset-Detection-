# cpu_augmentation.py
from torchvision import transforms

def get_cpu_augmentations(is_train=True, target_size=(320, 320)):
    """
    Defines a lightweight augmentation pipeline that runs well on CPU.
    """
    if is_train:
        # For training, add some basic augmentations
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(target_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # For validation/testing, just resize and normalize
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(target_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_thermal_augmentations(target_size=(320, 320)):
    """
    Minimal transformations for single-channel thermal images.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size, antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Grayscale normalization
    ])