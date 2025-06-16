# verify_day2.py
import yaml
import torch
from memory_profiler import profile

# Import all our new Day 2 modules
from efficient_multimodal_dataset import EfficientMultimodalDataset
from mobilevit_backbone import create_mobilevit_backbone
from rt_detr_model import create_rt_detr_model
from simple_attention_viz import visualize_attention
import cv2

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@profile
def run_verification():
    """
    This function loads all components and runs a single forward pass
    to verify that data loading and model inference work correctly.
    Memory usage for this function will be profiled.
    """
    print("--- Day 2 Verification Script ---")
    config = load_config()

    # 1. Verify Multimodal Dataset Loading
    print("\n[1/4] Initializing Multimodal Dataset...")
    multimodal_dataset = EfficientMultimodalDataset(config, is_train=False)
    
    if len(multimodal_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # Fetch one sample
    rgb_tensor, thermal_tensor, targets = multimodal_dataset[0]
    print("✔ Successfully loaded one multimodal sample.")
    print(f"  - RGB Tensor Shape: {rgb_tensor.shape}")
    print(f"  - Thermal Tensor Shape: {thermal_tensor.shape}")

    # 2. Verify RT-DETR Model Inference
    print("\n[2/4] Initializing and running RT-DETR model...")
    rt_detr = create_rt_detr_model()
    # RT-DETR's predict method takes a source image, not just a tensor
    # We'll use a dummy image path for verification
    dummy_rgb_path = multimodal_dataset.rgb_base_dir + "/" + multimodal_dataset.images[multimodal_dataset.image_ids[0]]['file_name']
    results = rt_detr.predict(source=dummy_rgb_path, verbose=False)
    print("✔ RT-DETR model ran successfully.")
    print(f"  - Detected {len(results[0].boxes)} objects in the sample image.")

    # 3. Verify MobileViT Backbone
    print("\n[3/4] Initializing and running MobileViT backbone...")
    backbone = create_mobilevit_backbone()
    features = backbone(rgb_tensor.unsqueeze(0)) # Add batch dimension
    print("✔ MobileViT backbone ran successfully.")
    print("  - Output feature map shapes:")
    for i, f in enumerate(features):
        print(f"    - Stage {i}: {f.shape}")

    # 4. Verify Attention Visualization
    print("\n[4/4] Generating attention visualization...")
    # For visualization, we need a standard ViT, not just the backbone
    from mobile_vit import create_lightweight_vit
    viz_model = create_lightweight_vit('mobilevit_s', num_classes=8)
    
    # We need a clean image tensor without normalization for visualization
    from torchvision.transforms import ToTensor, Resize
    raw_img_path = multimodal_dataset.rgb_base_dir + "/" + multimodal_dataset.images[multimodal_dataset.image_ids[0]]['file_name']
    raw_img = cv2.imread(raw_img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    viz_tensor = Resize((224, 224))(ToTensor()(raw_img))

    print(f"[DEBUG] viz_model type: {type(viz_model)}")
    print(f"[DEBUG] viz_model repr: {viz_model}")
    attention_image = visualize_attention(viz_model, viz_tensor)
    if attention_image is not None:
        cv2.imwrite("day2_attention_map.jpg", attention_image)
        print("✔ Attention map saved to 'day2_attention_map.jpg'.")
    else:
        print("✘ Failed to generate attention map.")
        
    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    # To run memory profiling, execute this from your terminal:
    # python -m memory_profiler verify_day2.py
    run_verification()