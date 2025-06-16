import json
from pathlib import Path

root_dir = 'data/au-air/au-air-camera-pan-tilt-01-2021-10-13-11-20-42'
json_path = Path(root_dir) / 'annotations.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Top-level keys in {json_path}:", list(data.keys()))
if isinstance(data, dict):
    for k, v in data.items():
        print(f"Key: {k}, Type: {type(v)}, Example value: {str(v)[:200]}")
