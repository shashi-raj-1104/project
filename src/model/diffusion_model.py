import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class LatentDiffusionModel(nn.Module):
    """
    Latent diffusion model:
    - Uses pretrained VAE: images 1024×1024 → latents 128×128
    - Uses custom UNet in latent space
    """

    def __init__(self, unet_model, vae_name="stabilityai/sd-vae-ft-mse", num_steps=1000):
        super().__init__()
        # Load pretrained VAE and freeze it
        print("Loading pretrained VAE:", vae_name)
        self.vae = AutoencoderKL.from_pretrained(vae_name)
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor  # scale latents

        self.unet = unet_model
        self.num_steps = num_steps

        # Create noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.loss_fn = nn.MSELoss()

    def encode(self, img):
        # img: [B,3,1024,1024], values in [-1,1]
        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
        return latents * self.scaling_factor  # [B,4,128,128]

    def decode(self, latents):
        latents = latents / self.scaling_factor
        with torch.no_grad():
            img = self.vae.decode(latents).sample
        return img  # [B,3,1024,1024]

    def q_sample(self, z0, t, noise):
        # Compute z_t from z_0 by adding noise
        a_bar = self.alpha_bar[t].view(-1,1,1,1).to(z0.device)
        return torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise

    def forward(self, img):
        # Training: image → latent → noisy latent → noise prediction loss
        z0 = self.encode(img)
        B = z0.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=img.device).long()

        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)

        noise_pred = self.unet(zt, t)
        loss = self.loss_fn(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, zt, t):
        # Reverse single timestep
        noise_pred = self.unet(zt, t)
        a = self.alphas[t].view(-1,1,1,1).to(zt.device)
        a_bar = self.alpha_bar[t].view(-1,1,1,1).to(zt.device)

        mean = (1.0 / torch.sqrt(a)) * (zt - ((1 - a) / torch.sqrt(1 - a_bar)) * noise_pred)
        if t == 0:
            return mean
        noise = torch.randn_like(zt)
        sigma = torch.sqrt(self.betas[t]).view(-1,1,1,1).to(zt.device)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, shape, device):
        # shape example: (1,4,128,128)
        z = torch.randn(shape).to(device)
        for i in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            z = self.p_sample(z, t)
        img = self.decode(z)
        return img
