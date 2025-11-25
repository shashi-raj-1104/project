import sys
import os

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.setup import init
from src.run_inference import run_inference

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
# HF_TOKEN = "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"


# Paths
IMG_PATH = "data/image.png"
MASK_PATH = "data/mask.png"

OUTPUT_PATH = "outputs/result.png"

# Model config
CONFIG = {
    'model': {
        'pretrained': 'runwayml/stable-diffusion-inpainting'
    }
}

def main():
    init(HF_TOKEN)
    run_inference(IMG_PATH, MASK_PATH, OUTPUT_PATH, CONFIG)

if __name__ == "__main__":
    main()
