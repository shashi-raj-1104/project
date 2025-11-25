"""Evaluation metrics for satellite image inpainting"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(original, generated):
    """Calculate PSNR between original and generated images"""
    original_np = np.array(original)
    generated_np = np.array(generated)
    return psnr(original_np, generated_np, data_range=255)

def calculate_ssim(original, generated):
    """Calculate SSIM between original and generated images"""
    original_np = np.array(original).astype(np.float64)
    generated_np = np.array(generated).astype(np.float64)
    
    # Handle different channel configurations
    if original_np.ndim == 3:
        channel_axis = 2
    else:
        channel_axis = None
        
    return ssim(original_np, generated_np, 
                data_range=255, channel_axis=channel_axis)

def calculate_lpips(original, generated, model=None):
    """Calculate LPIPS (perceptual similarity)"""
    # This would require the lpips package
    return 0.0  # Placeholder