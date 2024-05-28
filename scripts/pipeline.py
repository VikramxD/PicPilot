import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from utils import (accelerator, ImageAugmentation, clear_memory)
import hydra
from omegaconf import DictConfig
from PIL import Image
import lightning.pytorch as pl
pl.seed_everything(1234)


class AutoPaintingPipeline:
    """
    AutoPaintingPipeline class represents a pipeline for auto painting using an inpainting model from diffusers.
    
    Args:
        model_name (str): The name of the pretrained inpainting model.
        image (Image): The input image to be processed.
        mask_image (Image): The mask image indicating the areas to be inpainted.
    """
    
    def __init__(self, model_name: str, image: Image, mask_image: Image):
        self.model_name = model_name
        self.device = accelerator()
        self.pipeline = AutoPipelineForInpainting.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.image = load_image(image)
        self.mask_image = load_image(mask_image)
        self.pipeline.to(self.device)
       
        
        
    def run_inference(self, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float):
        """
        Runs the inference on the input image using the inpainting pipeline.
        
        Returns:
            Image: The output image after inpainting.
        """
        clear_memory()
        image = load_image(self.image)
        mask_image = load_image(self.mask_image)
        output = self.pipeline(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image,num_inference_steps=num_inference_steps,strength=strength,guidance_scale=guidance_scale,height = 1472, width = 2560).images[0]
        
        return output
    
    
@hydra.main(version_base=None ,config_path="../configs", config_name="inpainting")
def inference(cfg: DictConfig):
    """
    Load the configuration file for the inpainting pipeline.
    
    Args:
        cfg (DictConfig): The configuration file for the inpainting pipeline.
    """
    augmenter = ImageAugmentation(target_width=cfg.target_width, target_height=cfg.target_height, roi_scale=cfg.roi_scale)
    model_name = cfg.model
    image_path = "../sample_data/example5.jpg"
    image = Image.open(image_path)
    extended_image = augmenter.extend_image(image)
    mask_image = augmenter.generate_mask_from_bbox(extended_image, cfg.segmentation_model, cfg.detection_model)
    mask_image = augmenter.invert_mask(mask_image)
    prompt = cfg.prompt
    negative_prompt = cfg.negative_prompt
    num_inference_steps = cfg.num_inference_steps
    strength = cfg.strength
    guidance_scale = cfg.guidance_scale
    pipeline = AutoPaintingPipeline(model_name=model_name, image=extended_image, mask_image=mask_image)
    output = pipeline.run_inference(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale)
    output.save(f'{cfg.output_path}/output.jpg')
    return output
    
if __name__ == "__main__":
    inference()

        

  