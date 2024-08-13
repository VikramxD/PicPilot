from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import torch
from PIL import Image
import lightning.pytorch as pl
from scripts.api_utils import accelerator
from typing import Optional
pl.seed_everything(42)

class ImageGenerator:
    """
    A class to generate images using ControlNet and Stable Diffusion XL pipelines.

    Attributes:
        controlnet (ControlNetModel): The ControlNet model.
        pipeline (StableDiffusionXLControlNetPipeline): The Stable Diffusion XL pipeline with ControlNet.
    """

    def __init__(self, controlnet_model_name, sd_pipeline_model_name):
        """
        Initializes the ImageGenerator with the specified models.

        Args:
            controlnet_model_name (str): The name of the ControlNet model.
            sd_pipeline_model_name (str): The name of the Stable Diffusion XL pipeline model.
            image (str): The path to the image to be used.
        """
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            sd_pipeline_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=self.controlnet,
        ).to(accelerator())

    def inference(self, prompt, negative_prompt, height, width, guidance_scale, num_images_per_prompt, num_inference_steps, image_path, controlnet_conditioning_scale, control_guidance_end,output_path:Optional[str]):
        """
        Generates images based on the provided parameters.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            height (int): The height of the generated images.
            width (int): The width of the generated images.
            guidance_scale (float): The guidance scale for image generation.
            num_images_per_prompt (int): The number of images to generate per prompt.
            num_inference_steps (int): The number of inference steps.
            image_path (str): The path to the image to be used.
            controlnet_conditioning_scale (float): The conditioning scale for ControlNet.
            control_guidance_end (float): The end guidance for ControlNet.

        Returns:
            list: A list of generated images.
        """
        images_list = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            image=Image.open(image_path),
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_end=control_guidance_end,
        ).images
        if output_path:
            for i,image in enumerate(images_list):
                 image.save(f'{output_path}/output_{i}.png')
        else:
            return images_list
        
if __name__ == "__main__":
    generator = ImageGenerator(
        controlnet_model_name="destitech/controlnet-inpaint-dreamer-sdxl",
        sd_pipeline_model_name="RunDiffusion/Juggernaut-XL-v9"
    )
    generator.inference(
        prompt='Park',
        negative_prompt='low Resolution , Bad Resolution',
        height=1080,
        width=1920,
        guidance_scale=7.5,
        num_images_per_prompt=4,
        num_inference_steps=100,
        image_path='/home/PicPilot/sample_data/example1.jpg',
        controlnet_conditioning_scale=0.9,
        control_guidance_end=0.9,
        output_path='/home/PicPilot/output'
    )
    
        
    
    
