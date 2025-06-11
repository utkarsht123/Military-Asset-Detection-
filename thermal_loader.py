import os
import cv2
import numpy as np

class ThermalImageLoader:
    """
    Loads and preprocesses thermal images for the HIT-UAV and AU-AIR datasets.
    """

    def __init__(self, image_folder, image_size=(320, 256), normalize=True):
        """
        Args:
            image_folder (str): Path to the folder containing thermal images.
            image_size (tuple): Desired output image size (width, height).
            normalize (bool): Whether to normalize pixel values to [0, 1].
        """
        self.image_folder = image_folder
        self.image_size = image_size
        self.normalize = normalize

    def list_images(self, exts=('.png', '.jpg', '.jpeg', '.tiff')):
        """
        Lists all image files in the image folder with given extensions.
        """
        files = []
        for fname in os.listdir(self.image_folder):
            if fname.lower().endswith(exts):
                files.append(os.path.join(self.image_folder, fname))
        return sorted(files)

    def load_image(self, image_path):
        """
        Loads a single thermal image and preprocesses it.
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        # Normalize
        if self.normalize:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    def load_dataset(self):
        """
        Loads and preprocesses all thermal images in the folder.
        Returns:
            images (list of np.ndarray): List of preprocessed images.
            image_paths (list of str): Corresponding image file paths.
        """
        image_paths = self.list_images()
        images = []
        for path in image_paths:
            img = self.load_image(path)
            images.append(img)
        return images, image_paths

def load_hit_uav_thermal(image_folder="data/images", annotation_file='data/annotations.json'):
    """
    Loads HIT-UAV thermal images and (optionally) annotations.
    Args:
        image_folder (str): Path to HIT-UAV thermal images.
        annotation_file (str): Path to annotation file (optional).
    Returns:
        dataset (list of dict): Each dict contains 'image', 'image_path', and optionally 'annotation'.
    """
    loader = ThermalImageLoader(image_folder)
    images, image_paths = loader.load_dataset()
    dataset = []
    annotations = None
    if annotation_file and os.path.exists(annotation_file):
        # Assume annotation file is in COCO or similar JSON format
        import json
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        # Map image filename to annotation
        ann_map = {}
        for ann in annotations.get('images', []):
            ann_map[ann['file_name']] = ann
    for img, path in zip(images, image_paths):
        entry = {'image': img, 'image_path': path}
        if annotations:
            fname = os.path.basename(path)
            if fname in ann_map:
                entry['annotation'] = ann_map[fname]
        dataset.append(entry)
    return dataset
