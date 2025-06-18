# attention_analysis_lite.py
import torch
import cv2
import numpy as np
from PIL import Image

def generate_backbone_attention_map(model, rgb_tensor, target_size=(320, 320)):
    """
    Generates an attention map from the backbone's final feature layer.
    """
    model.eval()
    
    # Get the feature map from the last stage of the backbone
    with torch.no_grad():
        feature_maps = model.rgb_backbone(rgb_tensor.unsqueeze(0))
        last_feature_map = feature_maps[-1].squeeze(0) # Remove batch dim

    # Average across the channel dimension to get a 2D heatmap
    heatmap = torch.mean(last_feature_map, 0).cpu().numpy()
    
    # Resize heatmap to target image size and normalize
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on the original image
    # Convert tensor to a displayable image
    rgb_image_pil = Image.fromarray((rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    rgb_image_cv2 = cv2.cvtColor(np.array(rgb_image_pil), cv2.COLOR_RGB2BGR)
    rgb_image_cv2 = cv2.resize(rgb_image_cv2, target_size)

    superimposed_img = cv2.addWeighted(rgb_image_cv2, 0.6, heatmap_color, 0.4, 0)
    
    return superimposed_img
    