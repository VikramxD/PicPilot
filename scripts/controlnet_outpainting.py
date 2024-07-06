from diffusers import ControlNetModel,StableDiffusionXLControlNetPipeline
import torch
import requests
from PIL import Image
from io import BytesIO


controlnet = ControlNetModel.from_pretrained(
    "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
)

response = requests.get("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/outpainting/313891870-adb6dc80-2e9e-420c-bac3-f93e6de8d06b.png?download=true")
control_image = Image.open('/home/PicPilot/sample_data/example2.jpg')


pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    torch_dtype=torch.float16,
    variant="fp16",
    controlnet=controlnet,
).to("cuda")

image = pipeline(
    prompt='Showcase 4k',
    negative_prompt='low Resolution , Bad Resolution',
    height=1024,
    width=1024,
    guidance_scale=7.5,
    num_inference_steps=100,
    image=control_image,
    controlnet_conditioning_scale=0.9,
    control_guidance_end=0.9,
).images[0]

image.save('output.png')