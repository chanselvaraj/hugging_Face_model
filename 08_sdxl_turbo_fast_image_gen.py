# model  : stabilityai/sdxl-turbo
# task   : ultra-fast image generation — same prompt at 1, 2, and 4 steps

import torch
import time
from diffusers import AutoPipelineForText2Image
from pathlib import Path

MODEL_NAME = "stabilityai/sdxl-turbo"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = (
    "A majestic golden dragon soaring above ancient Japanese cherry blossom trees, "
    "sunset sky, epic fantasy art, highly detailed scales, glowing eyes, "
    "Studio Ghibli inspired, vibrant colours"
)

# guidance_scale must be 0 for turbo — it was trained without classifier-free guidance
STEP_CONFIGS = [
    {"num_inference_steps": 1, "guidance_scale": 0.0},
    {"num_inference_steps": 2, "guidance_scale": 0.0},
    {"num_inference_steps": 4, "guidance_scale": 0.0},
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"loading on {device.upper()}:", MODEL_NAME)
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)

    print(f"\nprompt: {PROMPT}\n")

    for cfg in STEP_CONFIGS:
        steps = cfg["num_inference_steps"]
        start = time.perf_counter()
        result = pipe(prompt=PROMPT, generator=torch.Generator(device).manual_seed(42), **cfg)
        elapsed = time.perf_counter() - start
        path = OUTPUT_DIR / f"sdxl_turbo_{steps}step.png"
        result.images[0].save(path)
        print(f"  {steps} step(s) → {path}  ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
