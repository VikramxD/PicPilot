<<<<<<< HEAD
from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline,AutoPipelineForInpainting
import torch






class PipelineFetcher:
    """
    A class that fetches different pipelines for image processing.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image (str): The image to be processed.

    """

    def __init__(self, controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image: str):
        self.controlnet_adapter_model_name = controlnet_adapter_model_name
        self.controlnet_base_model_name = controlnet_base_model_name
        self.kandinsky_model_name = kandinsky_model_name
        self.image = image

    def ControlNetInpaintPipeline(self):
        """
        Fetches the ControlNet inpainting pipeline.

        Returns:
            pipe (StableDiffusionControlNetInpaintPipeline): The ControlNet inpainting pipeline.

        """
        controlnet = ControlNetModel.from_pretrained(self.controlnet_adapter_model_name, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.controlnet_base_model_name, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.to('cuda')

        return pipe

    def KandinskyPipeline(self):
        """
        Fetches the Kandinsky pipeline.

        Returns:
            pipe (AutoPipelineForInpainting): The Kandinsky pipeline.

        """
        pipe = AutoPipelineForInpainting.from_pretrained(self.kandinsky_model_name, torch_dtype=torch.float16)
        pipe.to('cuda')
        return pipe


def fetch_control_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image):
    """
    Fetches the control pipeline for image processing.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image: The input image for processing.

    Returns:
        pipe: The control pipeline for image processing.
    """
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    pipe = pipe_fetcher.ControlNetInpaintPipeline()
    return pipe


def fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image):
    """
    Fetches the Kandinsky pipeline.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image: The input image.

    Returns:
        pipe: The Kandinsky pipeline.
    """
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    pipe = pipe_fetcher.KandinskyPipeline()
    return pipe



=======
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from utils import (accelerator, ImageAugmentation, clear_memory)
import hydra
from omegaconf import OmegaConf, DictConfig
from PIL import Image
import lightning.pytorch as pl
pl.seed_everything(42)
generator = torch.Generator("cuda").manual_seed(92)

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
        self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
        
        
    def run_inference(self, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float):
        """
        Runs the inference on the input image using the inpainting pipeline.
        
        Returns:
            Image: The output image after inpainting.
        """
        
        image = load_image(self.image)
        mask_image = load_image(self.mask_image)
        output = self.pipeline(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image,num_inference_steps=num_inference_steps,strength=strength,guidance_scale =guidance_scale,height = 1472, width = 2560).images[0]
        clear_memory()
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
    pipeline = AutoPaintingPipeline(model_name=model_name, image=extended_image, mask_image=mask_image)
    output = pipeline.run_inference(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale)
    output.save(f'{cfg.output_path}/output.jpg')
    return output
    
if __name__ == "__main__":
    inference()

        

  
>>>>>>> a817fb6 (chore: Update .gitignore and add new files for inpainting pipeline)
