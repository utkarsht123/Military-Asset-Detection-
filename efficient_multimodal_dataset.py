# efficient_multimodal_dataset.py (Updated for Day 4)
import os
import cv2
import torch
from torch.utils.data import Dataset
from lightweight_sensor_parser import parse_annotations
from cpu_augmentation import get_cpu_augmentations, get_thermal_augmentations
from fast_thermal_alignment import get_thermal_image_list, get_aligned_thermal_path

def box_xywh_to_cxcywh(x):
    """Converts bounding box from [x, y, w, h] to [cx, cy, w, h] format."""
    x0, y0, w, h = x.unbind(-1)
    b = [(x0 + w / 2), (y0 + h / 2), w, h]
    return torch.stack(b, dim=-1)

class EfficientMultimodalDataset(Dataset):
    def __init__(self, config, is_train=True, mode='rgb'):
        self.config = config
        self.is_train = is_train
        self.mode = mode.lower()  # 'rgb' or 'thermal'

        # --- Setup for RGB Data (AU-AIR) ---
        rgb_root = config['data']['au_air_sampled_root']
        self.rgb_base_dir = rgb_root
        annotation_file = os.path.join(rgb_root, 'annotations.filtered.json')
        self.images, self.annotations, self.categories = parse_annotations(annotation_file, key_classes=None)
        self.image_ids = list(self.images.keys())
        self.cat_id_map = {og_id: i for i, og_id in enumerate(sorted(self.categories.keys()))}

        # --- Setup for Thermal Data (HIT-UAV) ---
        thermal_root = config['data']['hit_uav_root']
        self.thermal_image_list = get_thermal_image_list(thermal_root)
        self.thermal_id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in self.thermal_image_list}
        if self.mode == 'rgb':
            print("Dataset mode: RGB only (AU-AIR)")
        elif self.mode == 'thermal':
            print("Dataset mode: Thermal only (HIT-UAV)")
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}. Use 'rgb' or 'thermal'.")

        # --- Setup Augmentations ---
        self.rgb_transform = get_cpu_augmentations(is_train=self.is_train, target_size=(320, 320))
        self.thermal_transform = get_thermal_augmentations(target_size=(320, 320))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        annotations = self.annotations.get(image_id, [])
        target = {'boxes': [], 'labels': []}

        if self.mode == 'rgb':
            # --- RGB image from AU-AIR ---
            rgb_path = os.path.join(self.rgb_base_dir, img_info['file_name'])
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                print(f"Warning: Could not read RGB image at {rgb_path}. Skipping sample.")
                return self.__getitem__((idx + 1) % len(self))
            h, w, _ = rgb_image.shape
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            # Prepare targets
            for ann in annotations:
                box = torch.tensor(ann['bbox'])
                box = box_xywh_to_cxcywh(box)
                box /= torch.tensor([w, h, w, h], dtype=torch.float32)
                target['boxes'].append(box)
                target['labels'].append(self.cat_id_map[ann['category_id']])
            target['boxes'] = torch.stack(target['boxes']) if target['boxes'] else torch.zeros((0, 4))
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64) if target['labels'] else torch.zeros(0, dtype=torch.int64)
            rgb_tensor = self.rgb_transform(rgb_image)
            thermal_tensor = torch.zeros((1, rgb_tensor.shape[1], rgb_tensor.shape[2]), dtype=torch.float32)
        elif self.mode == 'thermal':
            # --- Thermal image from HIT-UAV ---
            tidx = idx % len(self.thermal_image_list)
            thermal_path = self.thermal_image_list[tidx]
            thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_image is None:
                print(f"Warning: Could not read thermal image at {thermal_path}. Skipping sample.")
                return self.__getitem__((idx + 1) % len(self))
            h, w = thermal_image.shape
            # Dummy targets (no annotation for HIT-UAV)
            target['boxes'] = torch.zeros((0, 4))
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            rgb_tensor = torch.zeros((3, h, w), dtype=torch.float32)
            thermal_tensor = self.thermal_transform(thermal_image)
        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        # Debug: print tensor shapes and dtypes
        print(f"RGB tensor shape: {rgb_tensor.shape}, dtype: {rgb_tensor.dtype}")
        print(f"Thermal tensor shape: {thermal_tensor.shape}, dtype: {thermal_tensor.dtype}")

        return rgb_tensor, thermal_tensor, target

def collate_fn(batch):
    """Custom collate function to handle variable-sized targets."""
    rgb_tensors, thermal_tensors, targets = zip(*batch)
    rgb_tensors = torch.stack(rgb_tensors, 0)
    thermal_tensors = torch.stack(thermal_tensors, 0)
    return rgb_tensors, thermal_tensors, targets