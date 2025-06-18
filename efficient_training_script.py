# efficient_training_script.py
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficient_multimodal_dataset import EfficientMultimodalDataset, collate_fn
from lightweight_multimodal_detr import LightweightMultiModalDETR
from essential_loss_functions import SetCriterion
from checkpoint_manager import save_checkpoint, load_checkpoint

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # --- 1. Setup ---
    config = load_config()
    device = torch.device(config['training']['device'])
    best_loss = float('inf')
    
    # --- 2. Data Loaders ---
    full_dataset = EfficientMultimodalDataset(config=config, is_train=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn # Use custom collate function
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        collate_fn=collate_fn
    )
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # --- 3. Model, Loss, and Optimizer ---
    num_classes = len(full_dataset.cat_id_map)
    model = LightweightMultiModalDETR(num_classes=num_classes).to(device)
    
    loss_weights = {'class': 1, 'bbox': 5, 'giou': 2}
    criterion = SetCriterion(num_classes=num_classes, eos_coef=0.1, loss_weights=loss_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(config['training']['epochs']):
        # -- Training --
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        for rgb, thermal, targets in progress_bar:
            rgb, thermal = rgb.to(device), thermal.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(rgb, thermal)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

        # -- Validation --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (rgb, thermal, targets) in enumerate(val_loader):
                print(f"Batch RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                print(f"Batch Thermal shape: {thermal.shape}, dtype: {thermal.dtype}")
                rgb, thermal = rgb.to(device), thermal.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(rgb, thermal)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")

        # -- Checkpoint --
        is_best = avg_val_loss < best_loss
        best_loss = min(avg_val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        }, is_best, filename='day4_checkpoint.pth.tar')

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()