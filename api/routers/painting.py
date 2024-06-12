import sys
sys.path.append("../scripts")
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from models.painting import InpaintingRequest
import uuid
from inpainting_pipeline import AutoPaintingPipeline
from utils import pil_to_s3_json, ImageAugmentation
from hydra import compose, initialize
import lightning.pytorch as pl
pl.seed_everything(42)

router = APIRouter()



#class InpaintingRequest(BaseModel):
   # prompt: str
   # negative_prompt: str
   # num_inference_steps: int
   # strength: float
   # guidance_scale: float

def augment_image(image, target_width, target_height, roi_scale, segmentation_model_name, detection_model_name):
    """
    Augments an image with a given prompt, model, and other parameters.

    Parameters:
    - image (str): The path to the image file.
    - target_width (int): The desired width of the augmented image.
    - target_height (int): The desired height of the augmented image.
    - roi_scale (float): The scale factor for the region of interest.

    Returns:
    - augmented_image (PIL.Image.Image): The augmented image.
    - inverted_mask (PIL.Image.Image): The inverted mask generated from the augmented image.
    """
    image = Image.open(image)
    image_augmentation = ImageAugmentation(target_width, target_height, roi_scale)
    image = image_augmentation.extend_image(image)
    mask = image_augmentation.generate_mask_from_bbox(image, segmentation_model_name, detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask

def run_inference(cfg: dict, image_path: str, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float):
    """
    Run inference using the provided configuration and input image.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
        image_path (str): Path to the input image file.
        prompt (str): Prompt for the inference process.
        negative_prompt (str): Negative prompt for the inference process.
        num_inference_steps (int): Number of inference steps to perform.
        strength (float): Strength parameter for the inference.
        guidance_scale (float): Guidance scale for the inference.

    Returns:
        dict: A JSON object containing the image ID and the signed URL.

    Raises:
        HTTPException: If an error occurs during the inference process.

    """
    image, mask_image = augment_image(image_path, 
                                      cfg['target_width'], 
                                      cfg['target_height'], 
                                      cfg['roi_scale'], 
                                      cfg['segmentation_model'], 
                                      cfg['detection_model'])
    
    pipeline = AutoPaintingPipeline(model_name=cfg['model'], 
                                    image=image, 
                                    mask_image=mask_image, 
                                    target_height=cfg['target_height'], 
                                    target_width=cfg['target_width'])
    output = pipeline.run_inference(prompt=prompt, 
                                    negative_prompt=negative_prompt, 
                                    num_inference_steps=num_inference_steps, 
                                    strength=strength, 
                                    guidance_scale=guidance_scale)
    return pil_to_s3_json(output, file_name="output.png")

@router.post("/kandinskyv2.2_inpainting")
async def inpainting_inference(image: UploadFile = File(...), 
                               prompt: str = "", 
                               negative_prompt: str = "", 
                               num_inference_steps: int = 50, 
                               strength: float = 0.5, 
                               guidance_scale: float = 7.5):
    """
    Run the inpainting/outpainting inference pipeline.

    Parameters:
    - image: UploadFile - The image file to be used for inpainting/outpainting.
    - prompt: str - The prompt text for guiding the inpainting/outpainting process.
    - negative_prompt: str - The negative prompt text for guiding the inpainting/outpainting process.
    - num_inference_steps: int - The number of inference steps to perform during the inpainting/outpainting process.
    - strength: float - The strength parameter for controlling the inpainting/outpainting process.
    - guidance_scale: float - The guidance scale parameter for controlling the inpainting/outpainting process.

    Returns:
    - result: The result of the inpainting/outpainting process.

    Raises:
    - HTTPException: If an error occurs during the inpainting/outpainting process.
    """
    try:
        image_bytes = await image.read()
        image_path = f"/tmp/{uuid.uuid4()}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        
        with initialize(version_base=None,config_path="../../configs"):
            cfg = compose(config_name="inpainting")

        result = run_inference(cfg, image_path, prompt, negative_prompt, num_inference_steps, strength, guidance_scale)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


