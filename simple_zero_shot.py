# simple_zero_shot.py
import torch
from tinyclip_integration import get_image_features, get_text_features

def run_zero_shot_classification(model, tokenizer, image_preprocessor, image_path, prompts_dict):
    """
    Performs zero-shot classification on an image given a dictionary of class prompts.
    """
    class_names = list(prompts_dict.keys())
    text_prompts = list(prompts_dict.values())
    
    # 1. Get features for the image and all text prompts
    image_features = get_image_features(model, image_preprocessor, image_path)
    text_features = get_text_features(model, tokenizer, text_prompts)
    
    # 2. Calculate cosine similarity
    # A higher value means the image and text are more similar.
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 3. Get the top prediction
    values, indices = similarity[0].topk(1)
    top_index = indices[0].item()
    
    predicted_class = class_names[top_index]
    confidence = values[0].item()
    
    return predicted_class, confidence