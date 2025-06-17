# checkpoint_manager.py
import torch
import os

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves model and training parameters at a checkpoint.
    'state' is a dict containing 'epoch', 'state_dict', 'optimizer' etc.
    """
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        torch.save(state, best_filename)
        print(f"Saved new best model to {best_filename}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Loads model and training parameters from a checkpoint."""
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer, start_epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return model, optimizer, 0