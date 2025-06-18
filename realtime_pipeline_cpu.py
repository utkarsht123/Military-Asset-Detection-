from PIL import Image
import torch
import numpy as np
from deployment_utils import load_model
from cpu_model_optimization import SimpleCNN

class InferencePipeline:
    def __init__(self, model_path):
        """
        Initialize the inference pipeline by loading the quantized model and setting up class mapping.
        """
        # Load the optimized INT8 model
        self.model = load_model(SimpleCNN, model_path, num_classes=3, is_quantized=True)
        self.model.eval()
        # Class mapping
        self.class_mapping = {0: 'Tank', 1: 'Helicopter', 2: 'Ship'}

    def _preprocess(self, pil_img):
        """
        Preprocess a PIL Image for model inference.
        - Resize to 32x32
        - Convert to tensor
        - Normalize to [0, 1]
        - Add batch dimension
        """
        # Resize
        img = pil_img.resize((32, 32))
        # Convert to numpy array and normalize
        img = np.array(img).astype(np.float32) / 255.0
        # If grayscale, convert to 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        # If image has alpha channel, remove it
        if img.shape[-1] == 4:
            img = img[..., :3]
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Convert to tensor
        tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
        return tensor

    def predict(self, pil_img):
        """
        Predict the class of a PIL image and return class name and confidences.
        """
        # Preprocess image
        tensor = self._preprocess(pil_img)
        # Run through model
        with torch.no_grad():
            logits = self.model(tensor)
            softmax = torch.nn.functional.softmax(logits, dim=1)
            confidences = softmax.cpu().numpy()[0]
        # Get predicted class index
        pred_idx = int(np.argmax(confidences))
        pred_class = self.class_mapping[pred_idx]
        # Build confidences dictionary
        conf_dict = {self.class_mapping[i]: float(confidences[i]) for i in range(len(confidences))}
        return {'prediction': pred_class, 'confidences': conf_dict} 