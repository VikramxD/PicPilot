from logger import rich_logger as l
from wandb.integration.diffusers import autolog
from config import Project_Name
from clear_memory import clear_memory
from typing import List
import numpy as np
import torch
from PIL import Image
from mask_generator import invert_mask
from diffusers.utils import load_image
from pipeline import fetch_control_pipeline,fetch_kandinsky_pipeline,fetch_kandinsky_prior_pipeline,fetch_kandinsky_img2img_pipeline
from config import controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name
import cv2 
import PIL.ImageOps
from transformers import pipeline



autolog(init=dict(project=Project_Name))






def make_controlnet_condition(image: Image.Image) -> Image.Image:
    """
    Applies image processing operations to create a controlnet condition image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The controlnet condition image.
    """
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def make_inpaint_condition(init_image, mask_image):
    """
    Prepare the initial image for inpainting by applying a mask.

    Args:
        init_image (PIL.Image.Image): The initial image.
        mask_image (PIL.Image.Image): The mask image.

    Returns:
        torch.Tensor: The prepared initial image for inpainting.

    Raises:
        AssertionError: If the image and mask have different sizes.

    """
    # Prepare control image
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image
    

def make_hint(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint


    





def controlnet_inpainting_inference(prompt,
                         image,
                         mask_image,
                         control_image,
                         num_inference_steps=200,
                         guidance_scale=1.2,
                         strength=5.0,
                         generator=torch.Generator(device="cpu").manual_seed(1)
                        ) -> List[Image.Image]:
    """
    Perform inpainting inference on an image using the given parameters.

    Args:
        prompt: The prompt for the inpainting inference.
        image: The input image to be inpainted.
        mask_image: The mask image indicating the regions to be inpainted.
        controlnet_conditioning_image: The conditioning image for the controlnet.
        num_inference_steps: The number of inference steps to perform (default: 200).
        guidance_scale: The scale factor for the guidance loss (default: 1.2).
        strength: The strength of the inpainting (default: 5.0).
        generator: The random number generator for reproducibility (default: torch.Generator(device="cpu").manual_seed(1)).

    Returns:
        A list of inpainted images.

    """
    clear_memory()
    pipe = fetch_control_pipeline(controlnet_adapter_model_name, controlnet_base_model_name,kandinsky_model_name, control_image)
    image = pipe(prompt = prompt,num_inference_steps=num_inference_steps, generator=generator, eta=1.0, image=image, mask_image=mask_image,guidance_scale=guidance_scale,strenght=strength, control_image=control_image).images[0]
    return image

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
    pipe = fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    output_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask_image).images[0]
    return output_image
def kandinsky_inpainting_inference(prompt,negative_prompt,image,mask_image):
    pipe = fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name,kandinsky_model_name, image)
    output_image = pipe(prompt=prompt,negative_prompt=negative_prompt,image=image,mask_image=mask_image).images[0]
    return output_image

def kandinsky_controlnet_inpainting_inference(prompt, negative_prompt, image, hint, generator=torch.Generator(device="cuda").manual_seed(43)):
    """
    Perform inpainting inference using the Kandinsky ControlNet model.

    Args:
        prompt (str): The prompt for the inpainting process.
        negative_prompt (str): The negative prompt for the inpainting process.
        image (torch.Tensor): The input image for inpainting.
        hint (torch.Tensor): The hint for guiding the inpainting process.
        generator (torch.Generator, optional): The random number generator. Defaults to CUDA generator with seed 43.

    Returns:
        torch.Tensor: The inpainted image.

    """
    prior_pipe = fetch_kandinsky_prior_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    img_embed = prior_pipe(prompt=prompt, image=image, strength=0.85, generator=generator)
    negative_embed = prior_pipe(prompt=negative_prompt, image=image, strength=1, generator=generator)
    controlnet_pipe = fetch_kandinsky_img2img_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    image = controlnet_pipe(image=image, strength=0.5, image_embeds=img_embed.image_embeds, negative_image_embeds=negative_embed.image_embeds, hint=hint, num_inference_steps=50, generator=generator, height=768, width=768).images[0]
    return image

  

if __name__ == '__main__':
    l.info("Kandinsky Inpainting Inference")
    image = load_image('/home/product_diffusion_api/sample_data/example2.jpg')
    image = image.resize((768, 768))
    mask_image = load_image('/home/product_diffusion_api/scripts/invert_mask.jpg')
    mask_image = mask_image.resize((768,768))
    prompt = "Product in a GYM 8k ultrarealistic "
    negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    output_image = kandinsky_inpainting_inference(prompt,negative_prompt,image,mask_image)
    output_image=output_image.resize((768,768))
    depth_estimator = pipeline("depth-estimation")
    hint = make_hint(output_image, depth_estimator).unsqueeze(0).half().to("cuda")
    final_output_image = kandinsky_controlnet_inpainting_inference(prompt,negative_prompt,image, hint)
    
    
    
    
    
   
    
    
    

