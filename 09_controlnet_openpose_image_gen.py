# model  : lllyasviel/control_v11p_sd15_openpose + runwayml/stable-diffusion-v1-5
# task   : pose-guided image generation — extract skeleton, redraw as new character

import torch
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Usain_Bolt_Olympics_2012.jpg/402px-Usain_Bolt_Olympics_2012.jpg"

PROMPT = (
    "A powerful warrior in gleaming medieval armour standing heroically, "
    "fantasy concept art, cinematic lighting, highly detailed, 4k"
)
NEGATIVE_PROMPT = "blurry, low quality, deformed limbs, watermark, text, ugly"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("step 1 — extracting pose from reference image")
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    ref = Image.open(BytesIO(requests.get(REFERENCE_IMAGE_URL, timeout=10).content)).convert("RGB").resize((512, 512))
    pose_image = pose_detector(ref)
    pose_image.save(OUTPUT_DIR / "controlnet_pose_skeleton.png")
    print(f"  skeleton saved → {OUTPUT_DIR / 'controlnet_pose_skeleton.png'}")

    print("\nstep 2 — generating new character in that pose")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL, controlnet=controlnet, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=pose_image,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=torch.Generator(device).manual_seed(42),
    )

    out_path = OUTPUT_DIR / "controlnet_warrior_pose.png"
    result.images[0].save(out_path)
    print(f"  result saved → {out_path}")


if __name__ == "__main__":
    main()
