# cpu_train_pipeline.py
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from efficient_data_loaders import AuAirDataset
from mobile_vit import create_lightweight_vit
from simple_evaluation import get_basic_metrics
from tensorboard_tracker import TensorboardTracker

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, data_loader, optimizer, device, accumulation_steps, tracker, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}")
    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Scale the loss for accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (only after accumulating gradients)
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps # Un-scale for logging
        progress_bar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
        
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
    tracker.log_scalar("Loss/train", avg_loss, epoch)

def evaluate(model, data_loader, device, tracker, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    metrics = get_basic_metrics(all_labels, all_preds)
    print(f"Epoch {epoch+1} Evaluation Metrics: {metrics}")
    tracker.log_scalar("Accuracy/eval", metrics['accuracy'], epoch)
    tracker.log_scalar("F1-score/eval", metrics['f1_score'], epoch)


def main():
    # --- 1. Setup ---
    config = load_config()
    device = torch.device(config['training']['device'])
    tracker = TensorboardTracker()

    # --- 2. Data Loaders ---
    # For this initial test, we'll use a 10% sample of our 20% data for quick evaluation
    full_dataset = AuAirDataset(root_dir=config['data']['au_air_sampled_root'], config=config)
    
    # Create a 90/10 train/val split from the sampled data
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # Create memory-efficient data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=2, # Use a couple of workers if your CPU can handle it
        pin_memory=True if device != 'cpu' else False
    )
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    # --- 3. Model ---
    model = create_lightweight_vit(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # --- 4. Training Pipeline ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    for epoch in range(config['training']['epochs']):
        train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            config['training']['accumulation_steps'],
            tracker,
            epoch
        )
        evaluate(model, val_loader, device, tracker, epoch)
        
    tracker.close()
    print("Training finished!")
    # Optional: Save the initial model
    torch.save(model.state_dict(), "initial_mobilevit_cpu.pth")

if __name__ == '__main__':
    main()