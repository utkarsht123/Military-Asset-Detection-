# tinyclip_integration.py
import torch
import open_clip
from PIL import Image

def load_compact_clip_model(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """
    Loads a compact, pretrained CLIP model from open_clip.
    'ViT-B-32' is a good balance of performance and size.
    """
    print(f"Loading CLIP model: {model_name} with pretrained weights: {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    print("CLIP model, preprocessor, and tokenizer loaded successfully.")
    return model, preprocess, tokenizer

def get_image_features(model, image_preprocessor, image_path):
    """Encodes a single image into a feature vector."""
    image = Image.open(image_path).convert("RGB")
    image_input = image_preprocessor(image).unsqueeze(0)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def get_text_features(model, tokenizer, text_prompts):
    """Encodes a list of text prompts into feature vectors."""
    text_inputs = tokenizer(text_prompts)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features