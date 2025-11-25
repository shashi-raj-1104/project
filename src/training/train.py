import torch
from torch.utils.data import DataLoader
from src.data.dataset import DeepGlobeDataset
from src.model.unet import SimpleUNet
from src.model.diffusion_model import LatentDiffusionModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and loader
    train_ds = DeepGlobeDataset(root_dir="data", split="train", img_size=1024)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Initialize model components
    unet = SimpleUNet(in_channels=4, base_channels=64).to(device)
    model = LatentDiffusionModel(unet_model=unet, num_steps=1000).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for img in train_dl:
            img = img.to(device)
            loss = model(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dl)
        print(f"Epoch {epoch}/{num_epochs} — Training Loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        if epoch in {10, 20, 35, 45, 50}:
            torch.save(unet.state_dict(),
                    f"checkpoints/checkpoint_unet_epoch{epoch}.pth")
            print(f"✅ Saved checkpoint at epoch {epoch}")

if __name__ == "__main__":
    train()
