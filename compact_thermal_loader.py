# compact_thermal_loader.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CompactThermalDataset(Dataset):
    def __init__(self, root_dir, target_size=(320, 320)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize for single-channel
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # Load as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Downsample efficiently
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        if self.transform:
            image = self.transform(image)
            
        # For now, return a dummy label
        label = torch.tensor(0)
        
        return image, label