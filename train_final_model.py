# train_final_model.py
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

from efficient_multimodal_dataset import EfficientMultimodalDataset, collate_fn
from lightweight_multimodal_detr import LightweightMultiModalDETR
from essential_loss_functions import DetectionLoss
from early_stopping import EarlyStopping
from tensorboard_tracker import TensorboardTracker

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, criterion, data_loader, optimizer, device, tracker, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for rgb, thermal, targets in progress_bar:
        rgb, thermal = rgb.to(device), thermal.to(device)
        optimizer.zero_grad()
        outputs = model(rgb, thermal)
        loss, loss_ce, loss_bbox = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{total_loss / (progress_bar.n + 1):.4f}"})
        
    avg_loss = total_loss / len(data_loader)
    tracker.log_scalar("Loss/train", avg_loss, epoch)

def validate_one_epoch(model, criterion, data_loader, device, tracker, epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for rgb, thermal, targets in progress_bar:
            rgb, thermal = rgb.to(device), thermal.to(device)
            outputs = model(rgb, thermal)
            loss, loss_ce, loss_bbox = criterion(outputs, targets)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': f"{total_loss / (progress_bar.n + 1):.4f}"})
            
    avg_loss = total_loss / len(data_loader)
    tracker.log_scalar("Loss/val", avg_loss, epoch)
    return avg_loss

def main():
    config = load_config()
    device = torch.device(config['training']['device'])
    tracker = TensorboardTracker(log_dir='./runs/final_model_training')
    
    # --- Data ---
    full_dataset = EfficientMultimodalDataset(config)
    num_classes = full_dataset.num_classes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # --- Model, Loss, Optimizer ---
    model = LightweightMultiModalDETR(num_classes=num_classes).to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    early_stopper = EarlyStopping(patience=5, verbose=True, path='best_model_checkpoint.pth')
    
    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        train_one_epoch(model, criterion, train_loader, optimizer, device, tracker, epoch)
        val_loss = validate_one_epoch(model, criterion, val_loader, device, tracker, epoch)
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            
    tracker.close()
    print("Training finished. Best model saved to 'best_model_checkpoint.pth'.")

if __name__ == '__main__':
    main()