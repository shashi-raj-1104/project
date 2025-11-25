import torch
from PIL import Image
import numpy as np

from src.model.unet import SimpleUNet
from src.model.diffusion_model import LatentDiffusionModel
from src.training.utils import denorm

def generate_image(checkpoint_path: str, latent_shape=(1, 4, 128, 128), device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Load UNet and diffusion model
    unet = SimpleUNet(in_channels=4, base_channels=64).to(device)
    model = LatentDiffusionModel(unet_model=unet, num_steps=1000).to(device)

    # Load trained UNet checkpoint (only UNet parameters)
    unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
    unet.eval()

    # Generate image
    with torch.no_grad():
        latent_noise = torch.randn(latent_shape).to(device)
        generated_img = model.generate(latent_shape).cpu()

    # Denormalize and save
    img_tensor = denorm(generated_img[0]).permute(1, 2, 0).numpy()
    img_uint8 = (img_tensor * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save("generated_image.png")
    print("✅ Generated image saved as generated_image.png")

if __name__ == "__main__":
    checkpoint = "checkpoints/checkpoint_unet_epoch50.pth"  # adjust path as needed
    generate_image(checkpoint)
