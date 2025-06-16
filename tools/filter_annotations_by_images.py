import os
import json

import pathlib

import os
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'au-air' / 'au-air-sampled'
ANNOTATION_PATH = str(DATA_DIR / 'annotations.json')
IMAGES_DIR = str(DATA_DIR / 'images')
OUTPUT_PATH = str(DATA_DIR / 'annotations.filtered.json')

def main():
    print(f"[DEBUG] ANNOTATION_PATH: {ANNOTATION_PATH}")
    print(f"[DEBUG] IMAGES_DIR: {IMAGES_DIR}")
    print(f"[DEBUG] OUTPUT_PATH: {OUTPUT_PATH}")
    with open(ANNOTATION_PATH, 'r') as f:
        ann = json.load(f)

    # Get all .jpg basenames actually present in the images directory
    available_images = set([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')])
    print(f"[DEBUG] Found {len(available_images)} .jpg files in images dir")

    # Map from basename to full image dict in original annotation
    basename_to_img = {os.path.basename(img['file_name']): img for img in ann['images']}

    # Collect only those image dicts that match actual files
    filtered_images = [basename_to_img[img_name] for img_name in available_images if img_name in basename_to_img]
    filtered_image_ids = set(img['id'] for img in filtered_images)
    print(f"[DEBUG] Matched {len(filtered_images)} images from annotation")

    # Filter annotations: keep only those whose image_id is in filtered_image_ids
    filtered_annotations = [a for a in ann['annotations'] if a.get('image_id', a.get('image_name')) in filtered_image_ids]
    print(f"[DEBUG] Matched {len(filtered_annotations)} annotations")

    # Copy all other top-level fields as-is, but replace images/annotations
    new_ann = {k: v for k, v in ann.items() if k not in ['images', 'annotations']}
    new_ann['images'] = filtered_images
    new_ann['annotations'] = filtered_annotations

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(new_ann, f, indent=2)
    print(f"Filtered annotation file written to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
