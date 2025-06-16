# tensorboard_tracker.py
from torch.utils.tensorboard import SummaryWriter

class TensorboardTracker:
    def __init__(self, log_dir='./runs/lightweight_exp'):
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def close(self):
        self.writer.close()