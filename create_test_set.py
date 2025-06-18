# create_test_set.py
import os
import json
import random
import shutil
from pathlib import Path

def split_data(train_ratio=0.8, annotation_file='data/au-air/au-air-sampled/annotations.filtered.json'):
    annotations_path = Path(annotation_file)
    annotation_dir = annotations_path.parent
    
    # Define output directories relative to annotation_dir
    trainval_dir = annotation_dir.parent / "au-air-trainval"
    test_dir = annotation_dir.parent / "au-air-test"
    
    print(f"Loading annotations from {annotations_path} ...")
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    trainval_images = images[:split_idx]
    test_images = images[split_idx:]
    
    print(f"Splitting data: {len(trainval_images)} for train/val, {len(test_images)} for test.")

    # Process a subset (train or test)
    def process_subset(subset_images, output_dir):
        output_img_dir = output_dir / "images"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        
        for img_info in subset_images:
            # Source path uses the original filename from the sampled data
            source_img_path = annotation_dir / img_info['file_name']
            # Destination path remains flat
            dest_img_path = output_img_dir / Path(img_info['file_name']).name
            shutil.copy(source_img_path, dest_img_path)
        
        # Write new annotation file with only the subset of images
        new_data = {k: v for k, v in data.items() if k != 'images'}
        new_data['images'] = subset_images
        with open(output_dir / "annotations.json", 'w') as f:
            json.dump(new_data, f, indent=2)
        print(f"Saved {len(subset_images)} images to {output_dir}")

    process_subset(trainval_images, trainval_dir)
    process_subset(test_images, test_dir)
    print("Splitting complete!")

if __name__ == '__main__':
    # IMPORTANT: Run this only ONCE!
    # It splits your sampled data into permanent train/val and test sets.
    split_data()