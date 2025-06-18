# results_analysis_focused.py
import os
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image

# Import our project modules
from lightweight_multimodal_detr import LightweightMultiModalDETR
from efficient_multimodal_dataset import EfficientMultimodalDataset, collate_fn
from essential_visualization import plot_bounding_box, plot_confusion_matrix, plot_performance_summary
from attention_analysis_lite import generate_backbone_attention_map

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- Day 6: Results Analysis & Visualization ---")
    config = load_config()
    device = torch.device('cpu')
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failure_cases"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "success_cases"), exist_ok=True)
    
    # --- Load Test Data ---
    test_dataset = EfficientMultimodalDataset(config, is_train=False)
    # Create reverse mapping from index to class name
    cat_id_to_name = {v: k for k, v in test_dataset.categories.items()}
    class_names = [cat_id_to_name[i] for i in range(len(cat_id_to_name))]
    class_names.append("No Object")
    
    # --- Load Model ---
    model = LightweightMultiModalDETR(num_classes=test_dataset.num_classes)
    model.load_state_dict(torch.load('best_model_checkpoint.pth', map_location=device))
    model.to(device)
    model.eval()

    # --- Run Inference on Test Set ---
    all_preds = []
    all_gts = []
    
    print(f"Running inference on {len(test_dataset)} test samples...")
    for i in tqdm(range(len(test_dataset))):
        rgb_tensor, thermal_tensor, target = test_dataset[i]
        
        with torch.no_grad():
            outputs = model(rgb_tensor.unsqueeze(0), thermal_tensor.unsqueeze(0))
        
        # Get predicted class
        pred_class = outputs['pred_logits'].argmax(dim=-1).item()
        all_preds.append(pred_class)

        # Get ground truth class
        if len(target['labels']) > 0:
            gt_class = target['labels'][0].item()
        else:
            gt_class = test_dataset.num_classes # "No Object" class
        all_gts.append(gt_class)

        # --- Save some qualitative examples ---
        if i < 5 or (pred_class != gt_class and len(os.listdir(f'{output_dir}/failure_cases')) < 5):
            is_success = pred_class == gt_class
            folder = "success_cases" if is_success else "failure_cases"
            
            img_pil = Image.fromarray((rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            pred_box = outputs['pred_boxes'][0].cpu().numpy()
            true_box = target['boxes'][0].cpu().numpy() if len(target['boxes']) > 0 else None
            pred_label = class_names[pred_class]
            true_label = class_names[gt_class]

            img_with_boxes = plot_bounding_box(img_cv2, pred_box, true_box, pred_label, true_label)
            cv2.imwrite(f"{output_dir}/{folder}/sample_{i}.jpg", img_with_boxes)

            # Generate attention map for this sample
            attention_map = generate_backbone_attention_map(model, rgb_tensor)
            cv2.imwrite(f"{output_dir}/{folder}/sample_{i}_attention.jpg", attention_map)

    # --- Generate Quantitative Results ---
    print("\nGenerating quantitative analysis plots...")
    
    # 1. Performance Metrics
    accuracy = accuracy_score(all_gts, all_preds)
    precision = precision_score(all_gts, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_gts, all_preds, average='weighted', zero_division=0)
    
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}
    print(f"Overall Metrics: {metrics}")
    plot_performance_summary(metrics, save_path=f"{output_dir}/performance_summary.png")
    
    # 2. Confusion Matrix
    plot_confusion_matrix(all_gts, all_preds, class_names, save_path=f"{output_dir}/confusion_matrix.png")

    print("\n--- Analysis Complete! Results saved in 'results/' directory. ---")

if __name__ == '__main__':
    main()