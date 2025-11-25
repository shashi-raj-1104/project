"""Logging utilities for training"""
import logging
import json
from datetime import datetime

def setup_logging(log_dir="training_logs"):
    """Setup training logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def log_training_metrics(epoch, train_loss, val_loss, psnr, ssim):
    """Log training metrics to file"""
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss, 
        'psnr': psnr,
        'ssim': ssim,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_logs/metrics.json', 'a') as f:
        f.write(json.dumps(metrics) + '\n')