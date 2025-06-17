# verify_day3.py
import yaml
import torch
import torch.nn as nn

# --- Day 2 & 3 Imports ---
from efficient_multimodal_dataset import EfficientMultimodalDataset
from mobilevit_backbone import create_mobilevit_backbone
from lightweight_encoders import SimpleThermalEncoder
from early_fusion_modules import ConcatenationFusion
from tinyclip_integration import load_compact_clip_model
from military_prompts_lite import get_military_prompts
from simple_zero_shot import run_zero_shot_classification

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class IntegratedFusionModel(nn.Module):
    """A wrapper model to demonstrate the end-to-end fusion pipeline."""
    def __init__(self):
        super().__init__()
        self.rgb_backbone = create_mobilevit_backbone()
        # The backbone outputs features from 4 stages. We'll take the last one.
        # For MobileViT-S, the last stage has 640 channels.
        rgb_feature_dim = self.rgb_backbone.feature_info.channels()[-1] 
        thermal_feature_dim = 256 # As defined in our encoder

        # We need a pooling layer to flatten the spatial features from the backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.thermal_encoder = SimpleThermalEncoder(output_dim=thermal_feature_dim)
        self.fusion_module = ConcatenationFusion(
            rgb_feature_dim=rgb_feature_dim,
            thermal_feature_dim=thermal_feature_dim,
            output_dim=512 # Final fused feature size
        )

    def forward(self, rgb_image, thermal_image):
        # Process RGB image
        # Note: backbone returns a list of feature maps from different stages
        rgb_features_maps = self.rgb_backbone(rgb_image)
        rgb_features_last_stage = rgb_features_maps[-1] # Get the last feature map
        rgb_pooled = self.pool(rgb_features_last_stage)
        rgb_flat = self.flatten(rgb_pooled)

        # Process Thermal image
        thermal_features = self.thermal_encoder(thermal_image)

        # Fuse features
        fused_output = self.fusion_module(rgb_flat, thermal_features)
        return fused_output

def run_verification():
    print("--- Day 3 Verification Script ---")
    config = load_config()

    # --- Part 1: Verify Early Fusion Architecture ---
    print("\n[1/2] Verifying the Early Fusion pipeline...")
    
    # 1. Load data
    dataset = EfficientMultimodalDataset(config, is_train=False)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot verify fusion.")
        return
    rgb_tensor, thermal_tensor, _ = dataset[0]
    # Add batch dimension for model input
    rgb_batch = rgb_tensor.unsqueeze(0)
    thermal_batch = thermal_tensor.unsqueeze(0)
    print("✔ Loaded one multimodal sample.")
    print(f"  - RGB Batch Shape: {rgb_batch.shape}")
    print(f"  - Thermal Batch Shape: {thermal_batch.shape}")
    
    # 2. Instantiate and run the integrated model
    fusion_model = IntegratedFusionModel()
    fusion_model.eval()
    with torch.no_grad():
        fused_features = fusion_model(rgb_batch, thermal_batch)
    print("✔ Fusion model ran successfully.")
    print(f"  - Final Fused Feature Vector Shape: {fused_features.shape}")


    # --- Part 2: Verify TinyCLIP and Zero-Shot Classification ---
    print("\n[2/2] Verifying the Zero-Shot Classification pipeline...")

    # 1. Load CLIP model and prompts
    clip_model, preprocessor, tokenizer = load_compact_clip_model()
    prompts = get_military_prompts()
    print(f"✔ Loaded CLIP and {len(prompts)} text prompts.")

    # 2. Get a sample image path
    image_id = dataset.image_ids[10] # Use a different image for fun
    img_info = dataset.images[image_id]
    image_path = f"{dataset.rgb_base_dir}/{img_info['file_name']}"
    print(f"  - Testing with image: {image_path}")

    # 3. Run zero-shot classification
    predicted_class, confidence = run_zero_shot_classification(
        clip_model, tokenizer, preprocessor, image_path, prompts
    )
    print("✔ Zero-shot classification ran successfully.")
    print(f"  - Prediction: '{predicted_class}' with confidence: {confidence:.2f}")

    print("\n--- Verification Complete ---")

if __name__ == '__main__':
    run_verification()