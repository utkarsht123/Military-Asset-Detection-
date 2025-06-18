import json
import os

ANNOTATION_PATH = os.path.join('data', 'au-air', 'au-air-sampled', 'annotations.filtered.json')
IMAGES_DIR = os.path.join('data', 'au-air', 'au-air-sampled', 'images')

def main():
    with open(ANNOTATION_PATH, 'r') as f:
        annotations = json.load(f)

    missing = []
    images = annotations.get('images', [])
    for img_info in images:
        rel_path = img_info['file_name']
        img_path = os.path.join('data', 'au-air', 'au-air-sampled', rel_path)
        if not os.path.isfile(img_path):
            missing.append(img_path)

    if missing:
        print(f"Missing {len(missing)} image files referenced in annotation:")
        for p in missing:
            print(p)
    else:
        print("All images referenced in the annotation file exist.")

if __name__ == '__main__':
    main()
