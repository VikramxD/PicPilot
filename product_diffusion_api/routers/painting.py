import sys
sys.path.append("../scripts")
from fastapi import APIRouter,File,UploadFile
from pydantic import BaseModel
import base64
from io import BytesIO
from typing import List
import uuid
from inpainting_pipeline import AutoPaintingPipeline
from functools import lru_cache
from s3_manager import S3ManagerService
from PIL import Image
import io
from scripts.utils import ImageAugmentation
import hydra
from omegaconf import DictConfig



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


def pil_to_s3_json(image: Image.Image, file_name) -> str:
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
    image: File(..., description="The image to be inpainted")
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    strength: float
    guidance_scale: float
    
    




def augment_image(image, target_width, target_height, roi_scale,segmentation_model_name,detection_model_name):
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
    mask = image_augmentation.generate_mask_from_bbox(image,segmentation_model_name,detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask


@hydra.main(version_base="1.3",config_path="conf",config_name="inpainting")
def run_inference(cfg,image,prompt,negative_prompt,num_inference_steps,strength,guidance_scale):
    image, mask_image = augment_image(image, cfg.width, cfg.height, cfg.roi_scale,cfg.segmentation_model_name,cfg.detection_model_name)
    pipeline = AutoPaintingPipeline(model_name=cfg.model_name, image = image, mask_image=mask_image, target_height=cfg.height, target_width=cfg.width)
    output = pipeline.run_inference(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale)
    pil_to_s3_json(output,file_name="output.jpg")
    
    
@router.post("/inpainting")
def inpainting_inference(request: InpaintingRequest):
    return run_inference(request)
