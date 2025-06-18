import torch
from torch.utils.data import DataLoader, TensorDataset
from cpu_model_optimization import SimpleCNN, quantize_model_int8
from deployment_utils import save_model

def create_calibration_dataloader(num_samples=200, batch_size=16):
    """
    Create a calibration dataloader with dummy data.
    
    Args:
        num_samples: Number of calibration samples
        batch_size: Batch size for the dataloader
        
    Returns:
        DataLoader: Calibration dataloader
    """
    # Create dummy images (3 channels, 32x32 pixels)
    dummy_images = torch.randn(num_samples, 3, 32, 32)
    dummy_labels = torch.randint(0, 3, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def main():
    """
    Create and save both FP32 and INT8 model variants.
    """
    print("=" * 60)
    print("CREATING MODEL VARIANTS")
    print("=" * 60)
    
    # Step 1: Instantiate the base FP32 model
    print("\n1. Creating FP32 model...")
    fp32_model = SimpleCNN(num_classes=3)
    print(f"✓ FP32 model created with {sum(p.numel() for p in fp32_model.parameters()):,} parameters")
    
    # Step 2: Create calibration data
    print("\n2. Creating calibration dataset...")
    calibration_dataloader = create_calibration_dataloader(num_samples=200, batch_size=16)
    print(f"✓ Calibration dataloader created: {len(calibration_dataloader)} batches")
    
    # Step 3: Create the quantized INT8 variant
    print("\n3. Creating INT8 quantized model...")
    int8_model = quantize_model_int8(fp32_model, calibration_dataloader)
    print("✓ INT8 model created successfully")
    
    # Step 4: Save both models
    print("\n4. Saving model variants...")
    
    # Save FP32 model
    save_model(fp32_model, "models/fp32_model.pth")
    print("✓ FP32 model saved to models/fp32_model.pth")
    
    # Save INT8 model
    save_model(int8_model, "models/int8_model.pth")
    print("✓ INT8 model saved to models/int8_model.pth")
    
    # Step 5: Print confirmation and summary
    print("\n" + "=" * 60)
    print("MODEL VARIANTS CREATED SUCCESSFULLY")
    print("=" * 60)
    print("✓ FP32 Model: models/fp32_model.pth")
    print("✓ INT8 Model: models/int8_model.pth")
    print("\nBoth models are ready for deployment and comparison!")

if __name__ == "__main__":
    main() 