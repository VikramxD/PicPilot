import torch
from diffusers import AutoPipelineForInpainting,DiffusionPipeline
from diffusers.utils import load_image
from utils import (accelerator, ImageAugmentation, clear_memory)
import hydra
from omegaconf import DictConfig
from PIL import Image
from functools import lru_cache


@lru_cache(maxsize=1)
class AutoPaintingPipeline:
    """
    AutoPaintingPipeline class represents a pipeline for auto painting using an inpainting model from diffusers.
    
    Args:
        model_name (str): The name of the pretrained inpainting model.
        image (Image): The input image to be processed.
        mask_image (Image): The mask image indicating the areas to be inpainted.
    """
    
    def __init__(self, model_name: str, image: Image, mask_image: Image,target_width: int, target_height: int):
        self.model_name = model_name
        self.device = accelerator()
        self.pipeline = AutoPipelineForInpainting.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.image = load_image(image)
        self.mask_image = load_image(mask_image)
        self.target_width = target_width
        self.target_height = target_height
        self.pipeline.to(self.device)
        self.pipeline.unet = torch.compile(self.pipeline.unet,mode='max-autotune')
        
       
        
        
    def run_inference(self, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float):
        """
        Runs the inference on the input image using the inpainting pipeline.
        
        Returns:
            Image: The output image after inpainting.
        """
       
        image = load_image(self.image)
        mask_image = load_image(self.mask_image)
        output = self.pipeline(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image,num_inference_steps=num_inference_steps,strength=strength,guidance_scale=guidance_scale, height = self.target_height ,width = self.target_width).images[0]
        return output
    
    
@hydra.main(version_base=None ,config_path="../configs", config_name="inpainting")
def inference(cfg: DictConfig):
    """
    Load the configuration file for the inpainting pipeline.
    
    Args:
        cfg (DictConfig): The configuration file for the inpainting pipeline.
    """
    augmenter = ImageAugmentation(target_width=cfg.target_width, target_height=cfg.target_height)
    model_name = cfg.model
    image_path = "../sample_data/example3.jpg"
    image = Image.open(image_path)
    extended_image = augmenter.extend_image(image)
    mask_image = augmenter.generate_mask_from_bbox(extended_image, cfg.segmentation_model, cfg.detection_model)
    mask_image = augmenter.invert_mask(mask_image)
    prompt = cfg.prompt
    negative_prompt = cfg.negative_prompt
    num_inference_steps = cfg.num_inference_steps
    strength = cfg.strength
    guidance_scale = cfg.guidance_scale
    pipeline = AutoPaintingPipeline(model_name=model_name, image = extended_image, mask_image=mask_image, target_height=cfg.target_height, target_width=cfg.target_width)
    output = pipeline.run_inference(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale)
    output.save(f'{cfg.output_path}/output.jpg')
    mask_image.save(f'{cfg.output_path}/mask.jpg')
    
    
if __name__ == "__main__":
    inference()

        

  