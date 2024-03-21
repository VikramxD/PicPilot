from logger import rich_logger as l
from wandb.integration.diffusers import autolog
from config import Project_Name
from clear_memory import clear_memory
from typing import List
import numpy as np
import torch
from PIL import Image
from mask_generator import convert_to_numpy_array, generate_mask
from diffusers.utils import load_image
import cv2
from config import controlnet_adapter_model_name,controlnet_base_model_name
from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline
autolog(init=dict(project=Project_Name))






 




def make_inpaint_condition(init_image, mask_image):
        # Prepare control image
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
    




def make_image_controlnet(image,
                          mask_image,
                          controlnet_conditioning_image,
                          positive_prompt: str, negative_prompt: str,
                          seed: int = 2356132) -> List[Image.Image]:
    """Method to make image using controlnet
    Args:
        image (np.ndarray): input image
        mask_image (np.ndarray): mask image
        controlnet_conditioning_image (np.ndarray): conditioning image
        positive_prompt (str): positive prompt string
        negative_prompt (str): negative prompt string
        seed (int, optional): seed. Defaults to 2356132.
    Returns:
        List[Image.Image]: list of generated images
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_adapter_model_name, torch_dtype=torch.float32)
    pipe =  StableDiffusionControlNetInpaintPipeline.from_pretrained(
            controlnet_base_model_name, controlnet=controlnet, torch_dtype=torch.float32
        )
   

   
   
    
    image = pipe(prompt=positive_prompt,negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, control_image=controlnet_conditioning_image).images[0]


    return image

if __name__ == "__main__":
    init_image = load_image('/home/product_diffusion_api/sample_data/example1.jpg')
    mask_image = load_image('/home/product_diffusion_api/scripts/mask.jpg')
    controlnet_conditioning_image = make_inpaint_condition(init_image=init_image,mask_image=mask_image)
    result = make_image_controlnet(positive_prompt="Product used in kitchen 4k natural photography",negative_prompt="No artifcats",image=init_image,mask_image=mask_image,controlnet_conditioning_image=controlnet_conditioning_image)


