import torch
import os

def save_model(model, path):
    """
    Save a PyTorch model's state_dict to a file.
    
    Args:
        model: PyTorch model to save
        path: File path where to save the model
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model's state_dict
    torch.save(model.state_dict(), path)
    print(f"Model saved successfully to: {path}")

def load_model(model_class, path, *args, is_quantized=False, **kwargs):
    """
    Load a PyTorch model from a saved state_dict, with optional quantization support.
    
    Args:
        model_class: The model class (e.g., MyCNN)
        path: Path to the saved state_dict file
        *args: Positional arguments for model constructor
        is_quantized (bool): If True, prepare the model for quantization before loading weights
        **kwargs: Keyword arguments for model constructor
        
    Returns:
        model: Loaded model in eval mode
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Instantiate the model
    model = model_class(*args, **kwargs)

    if is_quantized:
        # Prepare for static quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    
    # Load the state_dict
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully from: {path}")
    return model

def get_model_size_mb(model_path):
    """
    Get the size of a saved model file in megabytes.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        float: Size of the model file in MB
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get file size in bytes
    size_bytes = os.path.getsize(model_path)
    
    # Convert to megabytes
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb 