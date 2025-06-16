# simple_attention_viz.py (Corrected)
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image

def visualize_attention(model, image_tensor, target_size=(224, 224)):
    """
    Generates an attention map visualization for a single image.
    This is a simplified example and may need adjustment for specific model architectures.
    """
    model.eval()
    
    attention_map = None
    def hook(module, input, output):
        nonlocal attention_map
        # output[0] gives us the tensor for the first item in the batch
        attention_map = output[0].detach()

    # Find a good layer to hook. The last block's attention is a good choice.
    hook_handle = None
    try:
        # For timm MobileViT/ByobNet, the last MobileVitBlock's first transformer block's attn
        hook_handle = model.stages[4][1].transformer[0].attn.register_forward_hook(hook)
    except Exception as e:
        print("Could not find 'model.stages[4][1].transformer[0].attn'. Please check model architecture.")
        print(f"Exception: {e}")
        return None

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0)) # Add batch dimension
    
    if hook_handle:
        hook_handle.remove()

    if attention_map is None:
        print("Could not extract attention map from hook.")
        return None
    
    # --- THIS IS THE KEY FIX ---
    # Original logic was averaging over the wrong dimension.
    # We need to average across the embedding dimension (dim=1) to get a score for each patch.
    # Input shape is (sequence_len, embedding_dim), e.g., (49, 192)
    # Output shape after mean(dim=1) will be (sequence_len,), e.g., (49,)
    if attention_map.ndim == 2:
        attention_map = torch.mean(attention_map, dim=1)
    else:
        print(f"Unexpected attention map dimension: {attention_map.ndim}. Cannot process.")
        return None
    # --- END OF FIX ---
    
    # Now, we reconstruct the 2D heatmap from the 1D patch scores
    side_length = int(np.sqrt(attention_map.shape[0]))
    if side_length * side_length != attention_map.shape[0]:
         print(f"Cannot form a square map from {attention_map.shape[0]} patches. Aborting visualization.")
         return None
    
    attention_heatmap = attention_map.reshape(side_length, side_length).cpu().numpy()
    
    # Resize to original image size
    attention_heatmap = cv2.resize(attention_heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize and convert to a color map
    heatmap = cv2.normalize(attention_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose on original image
    # Use the un-normalized tensor for visualization
    original_image = to_pil_image(image_tensor, mode='RGB').resize(target_size)
    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return superimposed_img