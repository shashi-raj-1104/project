import torch
from PIL import Image
import numpy as np
from src.model.unet import SimpleUNet
from src.model.diffusion_model import LatentDiffusionModel
from src.training.utils import denorm

def sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    unet = SimpleUNet(in_channels=4, base_channels=64).to(device)
    model = LatentDiffusionModel(unet_model=unet, num_steps=1000).to(device)

    # Load the trained UNet checkpoint (for example epoch 50)
    unet.load_state_dict(torch.load("checkpoint_epoch_50.pth", map_location=device))
    unet.eval()

    # Set latent shape corresponding to your VAE latent (for 1024→128 size)
    latent_shape = (1, 4, 128, 128)
    with torch.no_grad():
        generated = model.generate(latent_shape).cpu()

    img = denorm(generated[0]).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("generated_image.png")
    print("✅ Saved generated_image.png")

if __name__ == "__main__":
    sample()
