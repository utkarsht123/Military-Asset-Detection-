# lightweight_sensor_parser.py
import json
from collections import defaultdict

def parse_annotations(annotation_file, key_classes=None):
    """
    Parses a COCO-style annotation file.
    Returns:
        - images: A dictionary mapping image_id to image info.
        - annotations: A dictionary mapping image_id to a list of annotations.
        - categories: A dictionary mapping category_id to full category dict.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Handle categories as a list of strings (e.g., ['Human', 'Car', ...])
    if isinstance(data['categories'], list) and all(isinstance(cat, str) for cat in data['categories']):
        cat_name_to_id = {name: idx for idx, name in enumerate(data['categories'])}
        if key_classes:
            key_class_ids = {cat_name_to_id[name] for name in key_classes if name in cat_name_to_id}
            categories = {name: idx for name, idx in cat_name_to_id.items() if name in key_classes}
        else:
            key_class_ids = None
            categories = cat_name_to_id.copy()
    else:
        raise TypeError("Expected 'categories' to be a list of strings.")

    images = {img['id']: img for img in data['images']}
    
    # Filter annotations by key classes if specified
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        # Use 'class' key for category index
        ann_class = ann.get('class')
        # Some annotation entries may not have 'class'; skip if not present
        if ann_class is None:
            continue
        if key_class_ids is None or ann_class in key_class_ids:
            # The annotation may not have 'image_id', but likely has 'image_name'
            # Use image_name as the key if that's the convention
            image_key = ann.get('image_id', ann.get('image_name'))
            annotations_by_image[image_key].append(ann)
            
    return images, dict(annotations_by_image), categories