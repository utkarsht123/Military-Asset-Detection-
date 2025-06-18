# efficient_multimodal_dataset.py (Updated for Day 5)
import os
import cv2
import torch
from torch.utils.data import Dataset
from lightweight_sensor_parser import parse_annotations
from cpu_augmentation import get_cpu_augmentations, get_thermal_augmentations
from fast_thermal_alignment import get_thermal_image_list, get_aligned_thermal_path

class EfficientMultimodalDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.target_size = (320, 320)
        
        # --- Setup for RGB Data (AU-AIR) ---
        rgb_root = config['data']['au_air_sampled_root']
        self.rgb_base_dir = rgb_root
        annotation_file = os.path.join(rgb_root, 'annotations.filtered.json')
        self.images, self.annotations, self.categories = parse_annotations(annotation_file)
        self.image_ids = list(self.images.keys())
        # Create a mapping from COCO category_id to a 0-indexed integer
        self.cat_id_map = {og_id: i for i, og_id in enumerate(self.categories.keys())}
        self.num_classes = len(self.categories)

        # --- Setup for Thermal Data (HIT-UAV) ---
        thermal_root = config['data']['hit_uav_root']
        self.thermal_image_list = get_thermal_image_list(thermal_root)

        # --- Setup Augmentations ---
        self.rgb_transform = get_cpu_augmentations(is_train=self.is_train, target_size=self.target_size)
        self.thermal_transform = get_thermal_augmentations(target_size=self.target_size)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        rgb_path = os.path.join(self.rgb_base_dir, img_info['file_name'])
        rgb_image = cv2.imread(rgb_path)
        original_h, original_w, _ = rgb_image.shape
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # --- Load and Format Annotations ---
        annotations = self.annotations.get(image_id, [])
        boxes = []
        labels = []
        for ann in annotations:
            # Bbox format [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            # Normalize bounding box coordinates
            boxes.append([x / original_w, y / original_h, (x + w) / original_w, (y + h) / original_h])
            labels.append(self.cat_id_map[ann['category_id']])

        targets = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        # --- Load Thermal Image ---
        # Always cycle through HIT-UAV train thermal images
        thermal_train_dir = os.path.join(self.config['data']['hit_uav_root'], 'images', 'train')
        thermal_files = sorted([os.path.join(thermal_train_dir, f) for f in os.listdir(thermal_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not hasattr(self, '_thermal_idx'):
            self._thermal_idx = 0
        thermal_path = thermal_files[self._thermal_idx % len(thermal_files)]
        self._thermal_idx += 1
        thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_image is None:
            print(f"Warning: Could not read thermal image at {thermal_path}. Using zeros.")
            thermal_image = torch.zeros(self.target_size, dtype=torch.uint8).numpy()

        # Apply transformations
        rgb_tensor = self.rgb_transform(rgb_image)
        thermal_tensor = self.thermal_transform(thermal_image)

        return rgb_tensor, thermal_tensor, targets

# Custom collate function to handle variable-sized targets
def collate_fn(batch):
    rgb_images = torch.stack([item[0] for item in batch], 0)
    thermal_images = torch.stack([item[1] for item in batch], 0)
    targets = [item[2] for item in batch]
    return rgb_images, thermal_images, targets