# model  : runwayml/stable-diffusion-v1-5
# task   : text-to-image generation

import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = (
    "A futuristic cyberpunk city at night, neon lights reflecting on wet streets, "
    "flying cars weaving between skyscrapers, a lone detective in a trench coat "
    "standing under a glowing holographic billboard, ultra-detailed, cinematic lighting, "
    "8k resolution, photorealistic"
)

NEGATIVE_PROMPT = "blurry, low quality, deformed, bad anatomy, watermark, text, ugly, duplicate"

NUM_IMAGES = 2
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"loading on {device.upper()}:", MODEL_NAME)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    print(f"\nprompt: {PROMPT}\n")

    for i in range(NUM_IMAGES):
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device).manual_seed(42 + i),
        )
        path = OUTPUT_DIR / f"sd15_cyberpunk_{i + 1}.png"
        result.images[0].save(path)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
