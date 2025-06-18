# essential_loss_functions.py
import torch
import torch.nn.functional as F

class DetectionLoss(torch.nn.Module):
    def __init__(self, no_object_weight=0.1):
        super().__init__()
        self.no_object_weight = no_object_weight

    def forward(self, outputs, targets):
        # For our simplified model, we predict one box per image.
        # We'll compare it to the first ground truth box if it exists.
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # --- Prepare Targets ---
        # If a target exists, use its first box/label. Otherwise, use "no object".
        target_labels = []
        target_boxes = []
        
        for t in targets:
            if len(t['labels']) > 0:
                target_labels.append(t['labels'][0])
                # Convert [x_min, y_min, x_max, y_max] to [x_center, y_center, w, h]
                box = t['boxes'][0]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                w = box[2] - box[0]
                h = box[3] - box[1]
                target_boxes.append([cx, cy, w, h])
            else:
                # Use "no object" class index
                target_labels.append(pred_logits.shape[1] - 1) 
                target_boxes.append([0.0, 0.0, 0.0, 0.0]) # Dummy box

        target_labels = torch.as_tensor(target_labels, dtype=torch.long, device=pred_logits.device)
        target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32, device=pred_boxes.device)
        
        # --- Calculate Losses ---
        # 1. Classification Loss (Cross-Entropy)
        # Give less weight to the "no object" class to balance training
        class_weights = torch.ones(pred_logits.shape[1], device=pred_logits.device)
        class_weights[-1] = self.no_object_weight
        loss_ce = F.cross_entropy(pred_logits, target_labels, weight=class_weights)

        # 2. Bounding Box Loss (L1) - only for images with objects
        has_object_mask = target_labels != (pred_logits.shape[1] - 1)
        if has_object_mask.sum() > 0:
            loss_bbox = F.l1_loss(
                pred_boxes[has_object_mask], 
                target_boxes[has_object_mask], 
                reduction='mean'
            )
        else:
            loss_bbox = torch.tensor(0.0, device=pred_logits.device)
            
        total_loss = loss_ce + 5 * loss_bbox # Give more importance to bbox loss
        
        return total_loss, loss_ce, loss_bbox