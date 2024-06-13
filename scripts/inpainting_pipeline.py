import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from utils import accelerator, ImageAugmentation
import hydra
from omegaconf import DictConfig
from PIL import Image
from functools import lru_cache

@lru_cache(maxsize=1)
def load_pipeline(model_name: str, device, enable_compile: bool = True):
    pipeline = AutoPipelineForInpainting.from_pretrained(model_name, torch_dtype=torch.float16)
    if enable_compile:
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.unet = torch.compile(pipeline.unet, mode='reduce-overhead',fullgraph=True)
    pipeline.to(device)
    return pipeline

class AutoPaintingPipeline:
    def __init__(self, pipeline, image: Image, mask_image: Image, target_width: int, target_height: int):
        self.pipeline = pipeline
        self.image = image
        self.mask_image = mask_image
        self.target_width = target_width
        self.target_height = target_height

    def run_inference(self, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float):
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=self.image,
            mask_image=self.mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            height=self.target_height,
            width=self.target_width
            
        ).images[0]
        return output

@hydra.main(version_base=None, config_path="../configs", config_name="inpainting")
def inference(cfg: DictConfig):
    # Load the pipeline once and cache it
    pipeline = load_pipeline(cfg.model, accelerator(), True)

    # Image augmentation and preparation
    augmenter = ImageAugmentation(target_width=cfg.target_width, target_height=cfg.target_height)
    image_path = "../sample_data/example3.jpg"
    image = Image.open(image_path)
    extended_image = augmenter.extend_image(image)
    mask_image = augmenter.generate_mask_from_bbox(extended_image, cfg.segmentation_model, cfg.detection_model)
    mask_image = augmenter.invert_mask(mask_image)
    
    # Create AutoPaintingPipeline instance with cached pipeline
    painting_pipeline = AutoPaintingPipeline(
        pipeline=pipeline,
        image=extended_image,
        mask_image=mask_image,
        target_height=cfg.target_height,
        target_width=cfg.target_width
    )
    
    # Run inference
    output = painting_pipeline.run_inference(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.num_inference_steps,
        strength=cfg.strength,
        guidance_scale=cfg.guidance_scale
    )
    
    # Save output and mask images
    output.save(f'{cfg.output_path}/output.jpg')
    mask_image.save(f'{cfg.output_path}/mask.jpg')

if __name__ == "__main__":
    inference()

  