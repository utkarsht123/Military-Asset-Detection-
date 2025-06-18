# essential_visualization.py
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_bounding_box(image_cv2, pred_box=None, true_box=None, pred_label=None, true_label=None):
    """Draws predicted and ground truth boxes on an image."""
    h, w, _ = image_cv2.shape
    
    # Draw prediction (in green)
    if pred_box is not None:
        # Denormalize box: [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
        cx, cy, bw, bh = pred_box
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if pred_label:
            cv2.putText(image_cv2, f"Pred: {pred_label}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw ground truth (in blue)
    if true_box is not None:
        # Denormalize box: [x_min, y_min, x_max, y_max] -> itself
        x_min, y_min, x_max, y_max = true_box
        x_min, y_min = int(x_min * w), int(y_min * h)
        x_max, y_max = int(x_max * w), int(y_max * h)
        cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        if true_label:
            cv2.putText(image_cv2, f"GT: {true_label}", (x_min, y_max + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
    return image_cv2

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")

def plot_performance_summary(metrics_dict, save_path="results/performance_summary.png"):
    """Plots a bar chart of key performance metrics."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel("Score")
    plt.title("Key Performance Metrics")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
    plt.savefig(save_path, dpi=300)
    print(f"Performance summary chart saved to {save_path}")