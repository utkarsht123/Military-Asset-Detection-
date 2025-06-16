# efficient_data_loaders.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from lightweight_sensor_parser import parse_annotations

class AuAirDataset(Dataset):
    def __init__(self, root_dir, config, mode='train'):
        self.root_dir = root_dir
        self.config = config
        self.image_dir = os.path.join(root_dir, 'images')
        annotation_file = os.path.join(root_dir, 'annotations.json')
        
        key_classes = config['evaluation']['key_classes']
        self.images, self.annotations, self.categories = parse_annotations(annotation_file, key_classes)
        self.image_ids = list(self.images.keys())
        
        # A simple mapping from original category_id to a new 0-indexed id
        self.cat_id_map = {og_id: i for i, og_id in enumerate(sorted(self.categories.keys()))}

        # Lightweight transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # PyTorch expects RGB
        
        # For this basic pipeline, we'll simplify the task to image classification.
        # We'll assign the class of the first annotation to the image.
        # In later days, we will handle proper bounding boxes.
        annotations = self.annotations.get(image_id, [])
        if annotations:
            # Get the first annotation's category ID and map it
            category_id = annotations[0]['category_id']
            label = self.cat_id_map.get(category_id, 0) # Default to 0 if not found
        else:
            label = 0 # Assume a background class if no annotations

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)