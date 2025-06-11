import os
import numpy as np

from sensor_parser import load_auair_dataset, SensorDataPreprocessor
from thermal_loader import load_hit_uav_thermal

class MetricsCalculator:
    """
    Framework for calculating evaluation metrics for object detection and sensor data tasks.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_iou(boxA, boxB):
        """
        Computes Intersection over Union (IoU) between two bounding boxes.
        boxA, boxB: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
        return iou

    @staticmethod
    def mean_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        Computes mean Average Precision (mAP) for a single class.
        pred_boxes: list of [image_id, confidence, x1, y1, x2, y2]
        gt_boxes: list of [image_id, x1, y1, x2, y2]
        """
        pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
        image_gt = {}
        for gt in gt_boxes:
            image_id = gt[0]
            if image_id not in image_gt:
                image_gt[image_id] = []
            image_gt[image_id].append(gt[1:])

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        detected = {}

        for i, pred in enumerate(pred_boxes):
            image_id = pred[0]
            pred_box = pred[2:]
            max_iou = 0
            max_j = -1
            if image_id in image_gt:
                for j, gt_box in enumerate(image_gt[image_id]):
                    iou = MetricsCalculator.compute_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_j = j
            if max_iou >= iou_threshold:
                if (image_id, max_j) not in detected:
                    tp[i] = 1
                    detected[(image_id, max_j)] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (len(gt_boxes) + 1e-8)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t]
            if p.size > 0:
                ap += np.max(p)
        ap /= 11.0
        return ap

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))


def evaluate_auair(image_folder="data/images", annotation_file="data/annotations.json"):
    """
    Loads AU-AIR dataset and computes basic statistics.
    """
    dataset = load_auair_dataset(image_folder, annotation_file)
    print(f"Loaded {len(dataset)} AU-AIR samples.")
    # Example: count number of unique classes if 'category' in annotation
    categories = set()
    for item in dataset:
        if 'category' in item:
            categories.add(item['category'])
    print(f"Number of unique categories: {len(categories)}")
    return dataset

def evaluate_hit_uav(image_folder="data/images", annotation_file="data/annotations.json"):
    """
    Loads HIT-UAV thermal dataset and computes basic statistics.
    """
    dataset = load_hit_uav_thermal(image_folder, annotation_file)
    print(f"Loaded {len(dataset)} HIT-UAV thermal samples.")
    # Example: count how many have annotations
    annotated = sum(1 for item in dataset if 'annotation' in item)
    print(f"Samples with annotation: {annotated}")
    return dataset

def preprocess_sensor_data(sensor_data, missing_value_strategy='mean', normalize=True):
    """
    Preprocesses sensor data using SensorDataPreprocessor.
    """
    preprocessor = SensorDataPreprocessor(missing_value_strategy, normalize)
    return preprocessor.preprocess(sensor_data)

# Example usage (for demonstration; remove or adapt for actual experiments)
if __name__ == "__main__":
    # AU-AIR evaluation
    auair_data = evaluate_auair("data/images", "data/annotations.json")
    # HIT-UAV evaluation
    hit_uav_data = evaluate_hit_uav("data/images", "data/annotations.json")
    # Example: sensor data preprocessing
    # sensor_data = ... # Load your sensor data as list of dicts
    # processed = preprocess_sensor_data(sensor_data)
