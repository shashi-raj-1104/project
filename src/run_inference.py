import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def run_inference(image_path: str, mask_path: str, output_path: str, config: dict):
    """
    Run inpainting on a single image with a given mask.
    """
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config['model']['pretrained'],
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    output = pipe(
        prompt=[""] * 1,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        guidance_scale=7.5
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.images[0].save(output_path)
    print(f"✅ Result saved at {output_path}")
