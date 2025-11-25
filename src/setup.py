import os
import subprocess
from huggingface_hub import login

def init(hf_token: str):
    """
    Initialize environment: login, ensure output folder, install deps.
    """
    login(hf_token)

    os.makedirs("outputs", exist_ok=True)

    # You may already have some dependencies; these commands will ensure they’re installed.
    subprocess.run(["pip", "install", "-q",
                    "diffusers", "transformers", "accelerate",
                    "safetensors", "Pillow", "PyYAML", "torch"], check=False)
