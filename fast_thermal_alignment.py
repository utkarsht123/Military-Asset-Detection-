# fast_thermal_alignment.py
import os
import random

def get_thermal_image_list(thermal_root_dir):
    """Scans the thermal dataset directory and returns a list of image paths."""
    image_files = []
    # Assuming thermal images are in a subdirectory called 'images'
    thermal_img_dir = os.path.join(thermal_root_dir, 'images')
    if not os.path.exists(thermal_img_dir):
        print(f"Warning: Thermal image directory not found at {thermal_img_dir}")
        return []
        
    for f in os.listdir(thermal_img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(thermal_img_dir, f))
    return image_files

def get_aligned_thermal_path(rgb_image_info, thermal_image_list):
    """
    Simulates alignment by returning a random thermal image path.
    In a real system, this function would contain complex logic to find the
    corresponding thermal frame based on timestamp or other metadata.
    """
    if not thermal_image_list:
        return None # No thermal images available
    return random.choice(thermal_image_list)