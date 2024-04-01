from logger import rich_logger as l
from wandb.integration.diffusers import autolog
from config import Project_Name
from clear_memory import clear_memory
import numpy as np
import torch
from diffusers.utils import load_image,export_to_video
from pipeline import fetch_kandinsky_pipeline
from config import controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from video_pipeline import fetch_video_pipeline
from config import video_model_name








def make_inpaint_condition(init_image, mask_image):
    """
    Preprocesses the initial image and mask image for inpainting.

    Args:
        init_image (PIL.Image.Image): The initial image.
        mask_image (PIL.Image.Image): The mask image.

    Returns:
        torch.Tensor: The preprocessed image tensor.

    Raises:
        AssertionError: If the image and mask image have different sizes.
    """
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image
    


    







def kandinsky_inpainting_inference(prompt, negative_prompt, image, mask_image):
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
    pipe = fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name,kandinsky_model_name, image)
    output_image = pipe(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image,num_inference_steps=800,strength=1.0,guidance_scale = 7.8,height = 1472, width = 2560).images[0]
    return output_image

    


  

def sd2_inpainting_inference(prompt, img, mask, repo_id="stabilityai/stable-diffusion-2-inpainting", revision="fp16"):
    """
    Generate an image based on a prompt using a pretrained model.

    Args:
        prompt (str): The prompt for the image generation.
        img_url (str): The URL of the initial image.
        mask_url (str): The URL of the mask image.
        repo_id (str, optional): The ID of the repository of the pretrained model. Defaults to "stabilityai/stable-diffusion-2-inpainting".
        revision (str, optional): The revision of the pretrained model. Defaults to "fp16".

    Returns:
        Image: The generated image.
    """
    init_image = load_image(img)
    mask_image = load_image(mask)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=400,guidence_scale=7.5).images[0]
    return image

    
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



    
    
    
    

