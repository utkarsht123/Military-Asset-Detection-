# rt_detr_model.py
from ultralytics import RTDETR

def create_rt_detr_model(model_name='rtdetr-l.pt'):
    """
    Loads a pretrained RT-DETR model using the ultralytics library.
    'l' is for large, 'x' is for extra-large. We'll start with large.
    For a lighter version, you could use a custom-trained smaller one later.
    """
    print(f"Loading RT-DETR model: {model_name}")
    model = RTDETR(model_name)
    print("RT-DETR model loaded successfully.")
    return model