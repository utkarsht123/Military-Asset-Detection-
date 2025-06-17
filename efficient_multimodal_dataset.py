# efficient_multimodal_dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from lightweight_sensor_parser import parse_annotations
from cpu_augmentation import get_cpu_augmentations, get_thermal_augmentations
from fast_thermal_alignment import get_thermal_image_list, get_aligned_thermal_path

class EfficientMultimodalDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        
        # --- Setup for RGB Data (AU-AIR) ---
        rgb_root = config['data']['au_air_sampled_root']
        self.rgb_base_dir = rgb_root
        filtered_annotation_file = os.path.join(rgb_root, 'annotations.filtered.json')
        if os.path.exists(filtered_annotation_file):
            annotation_file = filtered_annotation_file
            print(f"[EfficientMultimodalDataset] Using filtered annotation file: {annotation_file}")
        else:
            annotation_file = os.path.join(rgb_root, 'annotations.json')
            print(f"[EfficientMultimodalDataset] Using original annotation file: {annotation_file}")
        self.images, self.annotations, _ = parse_annotations(annotation_file)
        # Only keep image_ids for which the image file actually exists
        valid_image_ids = []
        for image_id, img_info in self.images.items():
            rgb_path = os.path.join(self.rgb_base_dir, img_info['file_name'])
            if os.path.exists(rgb_path):
                valid_image_ids.append(image_id)
            else:
                print(f"[EfficientMultimodalDataset] Skipping missing image: {rgb_path}")
        self.image_ids = valid_image_ids

        # --- Setup for Thermal Data (HIT-UAV) ---
        thermal_root = config['data']['hit_uav_root']
        self.thermal_image_list = get_thermal_image_list(thermal_root)
        if not self.thermal_image_list:
            print("Warning: No thermal images loaded. The pipeline will run in RGB-only mode.")

        # --- Setup Augmentations ---
        self.rgb_transform = get_cpu_augmentations(is_train=self.is_train)
        self.thermal_transform = get_thermal_augmentations()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Load RGB Image and its annotations
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        rgb_path = os.path.join(self.rgb_base_dir, img_info['file_name'])
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # For Day 2, we pass labels as dummy data. We'll handle real labels in Day 4.
        # This focuses today on just getting data loaded and models running.
        targets = {'labels': [], 'boxes': []} # Placeholder for DETR format

        # 2. Find and Load Aligned Thermal Image
        thermal_path = get_aligned_thermal_path(img_info, self.thermal_image_list)
        if thermal_path:
            thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create a dummy black image if no thermal data is available
            h, w, _ = rgb_image.shape
            thermal_image = np.zeros((h, w), dtype=np.uint8)

        # 3. Apply Transformations
        # Note: We are not using PIL Images here to avoid an extra conversion step.
        # cv2 loads numpy arrays, which ToTensor() can handle directly.
        rgb_tensor = self.rgb_transform(rgb_image)
        thermal_tensor = self.thermal_transform(thermal_image)

        return rgb_tensor, thermal_tensor, targets