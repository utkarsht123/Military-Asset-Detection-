import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
import numpy as np

class SimpleCNN(nn.Module):
    """
    A simple CNN model for demonstration of INT8 quantization.
    This model will be used to test the quantization process.
    """
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 32x32 input -> 4x4 after 3 pooling layers (1024 features)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # First conv block
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))
        
        # Third conv block
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Dequantize output
        x = self.dequant(x)
        
        return x

    def predict(self, X):
        """
        Predict class indices for input tensor X.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Predicted class indices (batch_size,)
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            # Perform forward pass to get logits
            logits = self(X)
            
            # Get predicted class indices
            predictions = torch.argmax(logits, dim=1)
            
        return predictions

def quantize_model_int8(model, calibration_dataloader):
    """
    Perform INT8 static quantization on a PyTorch model.
    
    This function implements the complete quantization pipeline:
    1. Set model to evaluation mode
    2. Configure quantization backend
    3. Set quantization configuration
    4. Prepare model with observers
    5. Calibrate with calibration data
    6. Convert to quantized model
    
    Args:
        model: PyTorch model to quantize
        calibration_dataloader: DataLoader with calibration data (unlabeled)
        
    Returns:
        quantized_model: The quantized version of the input model
    """
    
    # Step 1: Set model to evaluation mode
    # Quantization requires the model to be in eval mode
    model.eval()
    print("✓ Model set to evaluation mode")
    
    # Step 2: Specify the backend for quantized engines
    # qnnpack is optimized for ARM processors, fbgemm for x86
    torch.backends.quantized.engine = 'qnnpack'
    print("✓ Quantization backend set to 'qnnpack'")
    
    # Step 3: Add quantization configuration to the model
    # This tells PyTorch how to quantize each layer
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print("✓ Quantization configuration added to model")
    
    # Step 4: Prepare the model by inserting observers
    # Observers will collect statistics about activations during calibration
    prepared_model = torch.quantization.prepare(model)
    print("✓ Observers inserted into model")
    
    # Step 5: Calibrate the model with calibration data
    # This step feeds data through the model to collect activation statistics
    print("Starting calibration...")
    with torch.no_grad():  # No gradients needed for calibration
        for batch_idx, (data, _) in enumerate(calibration_dataloader):
            # Feed data through the prepared model
            # The observers will record min/max values for each activation
            prepared_model(data)
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Calibrated batch {batch_idx}")
    
    print("✓ Calibration completed")
    
    # Step 6: Convert the observed model to a quantized model
    # This step actually quantizes the weights and activations to INT8
    quantized_model = torch.quantization.convert(prepared_model)
    print("✓ Model converted to INT8 quantized format")
    
    return quantized_model

def create_dummy_calibration_data(num_samples=100, batch_size=8):
    """
    Create dummy calibration data for testing quantization.
    
    Args:
        num_samples: Number of calibration samples to create
        batch_size: Batch size for the DataLoader
        
    Returns:
        calibration_dataloader: DataLoader with dummy data
    """
    # Create dummy images (3 channels, 32x32 pixels)
    dummy_images = torch.randn(num_samples, 3, 32, 32)
    dummy_labels = torch.randint(0, 3, (num_samples,))
    
    # Create a simple dataset
    dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def compare_model_sizes(original_model, quantized_model):
    """
    Compare the sizes of original and quantized models.
    
    Args:
        original_model: Original PyTorch model
        quantized_model: Quantized PyTorch model
        
    Returns:
        dict: Dictionary with size comparisons
    """
    # Calculate model sizes
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    compression_ratio = original_size / quantized_size
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - 1/compression_ratio) * 100
    }

def main():
    """
    Demo function to test the quantization process.
    """
    print("=" * 60)
    print("INT8 STATIC QUANTIZATION DEMO")
    print("=" * 60)
    
    # Create the demo model
    print("\n1. Creating SimpleCNN model...")
    model = SimpleCNN(num_classes=3)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy calibration data
    print("\n2. Creating calibration data...")
    calibration_dataloader = create_dummy_calibration_data(num_samples=100, batch_size=8)
    print(f"✓ Calibration data created: {len(calibration_dataloader)} batches")
    
    # Perform quantization
    print("\n3. Performing INT8 quantization...")
    quantized_model = quantize_model_int8(model, calibration_dataloader)
    print("✓ Quantization completed successfully!")
    
    # Compare model sizes
    print("\n4. Comparing model sizes...")
    size_comparison = compare_model_sizes(model, quantized_model)
    
    print(f"Original model size: {size_comparison['original_size_mb']:.2f} MB")
    print(f"Quantized model size: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"Size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    
    print("\n" + "=" * 60)
    print("QUANTIZATION DEMO COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 