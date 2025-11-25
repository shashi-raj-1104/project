import torch
import torch.nn.functional as F

def denorm(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor from [-1,1] to [0,1]
    """
    return (img.clamp(-1, 1) + 1) * 0.5

def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error between two image batches
    """
    return F.mse_loss(img1, img2, reduction='mean')

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two image batches.
    Assumes img1 and img2 are in [0,1] range.
    Formula: PSNR = 10 * log10( data_range^2 / MSE )
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(data_range * data_range / mse)
