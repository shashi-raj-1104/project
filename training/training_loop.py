# training/training.py
import torch
import torch.nn.functional as F

def train_diffusion(model, vae, dataloader, scheduler, optimizer, device):
    """
    training loop template.
    """
    model.train()
    for images in dataloader:
        images = images.to(device)

        # Encode image to latent space
        latents = vae.encode(images)

        # Sample timestep
        t = torch.randint(0, scheduler.timesteps, (images.size(0),), device=device)

        # Sample noise
        noise = torch.randn_like(latents)

        # Compute noisy latents
        beta = scheduler.get_beta(t).view(-1, 1, 1, 1)
        noisy_latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * noise

        # Predict noise
        pred_noise = model(noisy_latents, t)

        # Loss
        loss = F.mse_loss(pred_noise, noise)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
