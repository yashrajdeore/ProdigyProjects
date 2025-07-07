
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A futuristic car flying over a neon city, ultra realistic, 4k"
image = pipe(prompt).images[0]
image.save("output_image.png")
