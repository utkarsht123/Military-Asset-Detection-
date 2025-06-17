# fast_thermal_alignment.py
import os
import random

def get_thermal_image_list(thermal_root_dir):
    """
    Recursively scans the thermal dataset directory and returns a list of all image paths
    in all subdirectories (e.g., train, val, test) under 'images/'.
    """
    image_files = []
    thermal_img_dir = os.path.join(thermal_root_dir, 'images')
    if not os.path.exists(thermal_img_dir):
        print(f"Warning: Thermal image directory not found at {thermal_img_dir}")
        return []
    for root, _, files in os.walk(thermal_img_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, f))
    return image_files

def get_aligned_thermal_path(rgb_image_info, thermal_image_list):
    """
    Attempts to align a thermal image with the RGB image using filename matching.
    Falls back to random selection if no match is found.
    """
    if not thermal_image_list:
        return None  # No thermal images available

    # Try to match by filename (without extension)
    rgb_filename = os.path.splitext(os.path.basename(rgb_image_info['file_name']))[0]
    for t_path in thermal_image_list:
        t_base = os.path.splitext(os.path.basename(t_path))[0]
        if t_base == rgb_filename:
            return t_path
    # Fallback: random selection
    print(f"Warning: No aligned thermal image found for {rgb_filename}, using random thermal image.")
    return random.choice(thermal_image_list)