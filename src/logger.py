from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardLogger:
    def __init__(self, log_dir="runs/siamese_experiment"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log scalar values (loss, accuracy, etc.)"""
        self.writer.add_scalar(tag, value, step)

    def log_model_graph(self, model, sample_input):
        """Log model architecture"""
        self.writer.add_graph(model, sample_input)

    def close(self):
        """Close the writer"""
        self.writer.close()
