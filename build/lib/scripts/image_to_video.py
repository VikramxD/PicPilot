import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image 

pipeline = I2VGenXLPipeline.from_pretrained(
    "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image_url = (
    "/home/PicPilot/sample_data/image.png"
)
image = load_image(Image.open(image_url)).convert("RGB")
prompt = "Create a video from this Nike sneaker image,a zoom in shot , with a focus on the shoe at the center"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(42)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_frames = 60,
    num_inference_steps=30,
    negative_prompt=negative_prompt,
    guidance_scale=8.0,
    generator=generator,
    ).frames[0]
video_path = export_to_video(frames, "i2v.mp4")