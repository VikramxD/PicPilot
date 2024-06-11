import sys
sys.path.append("../scripts")
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
import uuid
from inpainting_pipeline import AutoPaintingPipeline
from s3_manager import S3ManagerService
from PIL import Image
import io
from utils import ImageAugmentation
from hydra import compose, initialize
import lightning.pytorch as pl
pl.seed_everything(42)

router = APIRouter()

def pil_to_b64_json(image):
    """
    Converts a PIL image to a base64-encoded JSON object.

    Args:
        image (PIL.Image.Image): The PIL image object to be converted.

    Returns:
        dict: A dictionary containing the image ID and the base64-encoded image.

    """
    image_id = str(uuid.uuid4())
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_id": image_id, "b64_image": b64_image}


def pil_to_s3_json(image: Image.Image, file_name) -> dict:
    """
    Uploads a PIL image to Amazon S3 and returns a JSON object containing the image ID and the signed URL.

    Args:
        image (PIL.Image.Image): The PIL image to be uploaded.
        file_name (str): The name of the file.

    Returns:
        dict: A JSON object containing the image ID and the signed URL.

    """
    image_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(image_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(
        unique_file_name, exp=43200
    )  # 12 hours
    return {"image_id": image_id, "url": signed_url}


class InpaintingRequest(BaseModel):
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    strength: float
    guidance_scale: float

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


