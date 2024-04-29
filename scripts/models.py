from logger import rich_logger as l
from wandb.integration.diffusers import autolog
from config import Project_Name
from clear_memory import clear_memory
import numpy as np
import torch
from diffusers.utils import load_image
from pipeline import fetch_kandinsky_pipeline
from config import controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from video_pipeline import fetch_video_pipeline
from config import video_model_name








    







def kandinsky_inpainting_inference(prompt, negative_prompt, image, mask_image,num_inference_steps=800,strength=1.0,guidance_scale = 7.8):
    """
    Perform Kandinsky inpainting inference on the given image.

    Args:
        prompt (str): The prompt for the inpainting process.
        negative_prompt (str): The negative prompt for the inpainting process.
        image (PIL.Image.Image): The input image to be inpainted.
        mask_image (PIL.Image.Image): The mask image indicating the areas to be inpainted.

    Returns:
        PIL.Image.Image: The output inpainted image.
    """
    clear_memory()
    l.info("Kandinsky Inpainting Inference ->")
    pipe = fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name,kandinsky_model_name, image)
    output_image = pipe(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image,num_inference_steps=num_inference_steps,strength=strength,guidance_scale = guidance_scale,height = 1472, width = 2560).images[0]
    return output_image

    


  



    
def image_to_video_pipeline(image, video_model_name, decode_chunk_size, motion_bucket_id, generator=torch.manual_seed(42)):
    """
    Converts an image to a video using a specified video model.

    Args:
        image (Image): The input image to convert to video.
        video_model_name (str): The name of the video model to use.
        decode_chunk_size (int): The size of the chunks to decode.
        motion_bucket_id (str): The ID of the motion bucket.
        generator (torch.Generator, optional): The random number generator. Defaults to torch.manual_seed(42).

    Returns:
        list: The frames of the generated video.
    """
    clear_memory()
    l.info("Stable Video Diffusion Image 2 Video pipeline Inference ->")
    pipe = fetch_video_pipeline(video_model_name)
    frames = pipe(image=image, decode_chunk_size=decode_chunk_size, motion_bucket_id=motion_bucket_id, generator=generator).frames[0]
    return frames



    
    
    
    

