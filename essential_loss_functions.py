# essential_loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    """Converts [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """Calculates GIoU loss."""
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    # Intersection area
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    C_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    C_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    C_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    C_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    C_area = (C_x2 - C_x1) * (C_y2 - C_y1)
    
    giou = iou - (C_area - union_area) / (C_area + 1e-6)
    return 1 - giou

class SetCriterion(nn.Module):
    """
    This class computes the loss for our DETR-like model.
    It combines a classification loss and a box regression loss.
    """
    def __init__(self, num_classes, eos_coef, loss_weights):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef # Weight for 'no-object' class
        self.loss_weights = loss_weights

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """
        Computes the loss for DETR.
        Args:
            outputs: dict of model outputs
            targets: list of dicts with 'boxes' and 'labels'
        Returns:
            total_loss: weighted sum of all losses
            losses: dict of individual loss components
        """
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        # NaN/Inf diagnostics
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print("[ERROR] pred_logits contains NaN or Inf!")
            print("pred_logits:", pred_logits)
            raise ValueError("pred_logits contains NaN or Inf!")
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            print("[ERROR] pred_boxes contains NaN or Inf!")
            print("pred_boxes:", pred_boxes)
            raise ValueError("pred_boxes contains NaN or Inf!")
        for i, t in enumerate(targets):
            for k, v in t.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"[ERROR] target[{i}]['{k}'] contains NaN or Inf! Value: {v}")
                    raise ValueError(f"target[{i}]['{k}'] contains NaN or Inf!")

        # --- Matcher: Find the best prediction for each ground truth box ---
        indices = []
        for i in range(pred_logits.shape[0]):
            out_prob = pred_logits[i].softmax(-1)
            out_box = pred_boxes[i]
            tgt_box = targets[i]["boxes"]
            tgt_label = targets[i]["labels"]

            cost_class = -out_prob[:, tgt_label]
            cost_bbox = torch.cdist(out_box, tgt_box, p=1)
            cost_giou = generalized_box_iou(out_box.unsqueeze(1), tgt_box.unsqueeze(0))
            
            cost_matrix = cost_bbox + self.loss_weights['class'] * cost_class + self.loss_weights['giou'] * cost_giou
            indices.append(linear_sum_assignment(cost_matrix.detach().cpu()))

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
        # --- Compute Losses ---
        # 1. Classification Loss (Cross Entropy)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o
        
        weight = torch.ones(self.num_classes + 1, device=pred_logits.device)
        weight[-1] = self.eos_coef # Lower weight for 'no-object'
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight)

        # 2. Box Regression Losses (L1 + GIoU)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / len(target_boxes)
        loss_giou = generalized_box_iou(src_boxes, target_boxes).sum() / len(target_boxes)

        # Final weighted loss
        losses = {
            'loss_class': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }
        total_loss = sum(losses[k] * self.loss_weights[k.split('_')[-1]] for k in losses.keys())
        return total_loss