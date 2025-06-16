import os
import json

# Paths
images_dir = r"C:\Users\Utkarsh\OneDrive\Desktop\new mili\data\au-air\au-air-sampled\images"
annotations_json = r"C:\Users\Utkarsh\OneDrive\Desktop\new mili\data\au-air\au-air-sampled\annotations.json"
filtered_json = r"C:\Users\Utkarsh\OneDrive\Desktop\new mili\data\au-air\au-air-sampled\annotations.filtered.json"

# Get list of image filenames (just the filename, not path)
image_filenames = set(os.listdir(images_dir))

# Helper to extract just the filename from the annotation's file_name
get_filename = lambda path: os.path.basename(path)

# Read and filter annotations
with open(annotations_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

filtered_images = []
for img_ann in data["images"]:
    filename = get_filename(img_ann["file_name"])
    if filename in image_filenames:
        filtered_images.append(img_ann)

# Construct new filtered json
filtered_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": filtered_images
}

with open(filtered_json, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4)
