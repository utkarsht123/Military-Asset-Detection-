import os
import json
from thermal_loader import load_hit_uav_thermal

class SensorDataPreprocessor:
    """
    Preprocess sensor data for AU-AIR and HIT-UAV datasets.
    """
    def __init__(self, missing_value_strategy='mean', normalize=True):
        self.missing_value_strategy = missing_value_strategy
        self.normalize = normalize

    def preprocess(self, sensor_data):
        """
        sensor_data: list of dicts (from csv.DictReader)
        Returns: list of dicts with preprocessed values
        """
        if not sensor_data:
            return []

        # Convert all numeric fields to float, handle missing values
        keys = sensor_data[0].keys()
        columns = {k: [] for k in keys}
        for row in sensor_data:
            for k in keys:
                val = row[k]
                if val == '' or val is None:
                    columns[k].append(None)
                else:
                    try:
                        columns[k].append(float(val))
                    except ValueError:
                        columns[k].append(val)  # keep as string if not convertible

        # Impute missing values
        for k in keys:
            col = columns[k]
            if all(isinstance(x, (float, int)) or x is None for x in col):
                if self.missing_value_strategy == 'mean':
                    valid = [x for x in col if x is not None]
                    mean_val = sum(valid) / len(valid) if valid else 0.0
                    columns[k] = [mean_val if x is None else x for x in col]
                elif self.missing_value_strategy == 'zero':
                    columns[k] = [0.0 if x is None else x for x in col]
                # Add more strategies as needed

        # Normalize numeric columns
        if self.normalize:
            for k in keys:
                col = columns[k]
                if all(isinstance(x, (float, int)) for x in col):
                    min_val = min(col)
                    max_val = max(col)
                    if max_val > min_val:
                        columns[k] = [(x - min_val) / (max_val - min_val) for x in col]
                    else:
                        columns[k] = [0.0 for _ in col]

        # Reconstruct list of dicts
        preprocessed = []
        for i in range(len(sensor_data)):
            row = {}
            for k in keys:
                row[k] = columns[k][i]
            preprocessed.append(row)
        return preprocessed

def load_auair_dataset(image_folder, annotation_file="data/annotations.json"):
    """
    Loads AU-AIR dataset annotations and matches them with image files.
    Returns: list of dicts with annotation and image path.
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    dataset = []
    for item in annotations:
        if isinstance(item, dict):
            image_name = item.get('image_name') or item.get('filename')
        elif isinstance(item, str):
            image_name = item
            item = {}  # or {'image_name': image_name}
        else:
            continue
        if not image_name:
            continue
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            entry = dict(item)
            entry['image_path'] = image_path
            dataset.append(entry)
    return dataset



