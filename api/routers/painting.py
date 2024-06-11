<<<<<<< HEAD
from pathlib import Path
import os
import uuid
from typing import List, Tuple, Any, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends, Body
from pydantic import BaseModel, Field
from PIL import Image
import lightning.pytorch as pl
from scripts.api_utils import pil_to_s3_json, pil_to_b64_json, ImageAugmentation, accelerator
from inpainting_pipeline import AutoPaintingPipeline, load_pipeline
from hydra import compose, initialize
from async_batcher.batcher import AsyncBatcher
import json
from functools import lru_cache

pl.seed_everything(42)
router = APIRouter()

# Initialize Hydra configuration
with initialize(version_base=None, config_path=Path(__file__).resolve().parent.parent / "configs"):
    cfg = compose(config_name="inpainting")

# Load the inpainting pipeline
@lru_cache(maxsize=1)
def load_pipeline_wrapper():
    """
    Load the inpainting pipeline with the specified configuration.

    Returns:
        pipeline: The loaded inpainting pipeline.
    """
    pipeline = load_pipeline(cfg.model, accelerator(), enable_compile=True)
    return pipeline
inpainting_pipeline = load_pipeline_wrapper()

class InpaintingRequest(BaseModel):
    """
    Model representing a request for inpainting inference.
    """
    prompt: str = Field(..., description="Prompt text for inference")
    negative_prompt: str = Field(..., description="Negative prompt text for inference")
    num_inference_steps: int = Field(..., description="Number of inference steps")
    strength: float = Field(..., description="Strength of the inference")
    guidance_scale: float = Field(..., description="Guidance scale for inference")
    mode: str = Field(..., description="Mode for output ('b64_json' or 's3_json')")
    num_images: int = Field(..., description="Number of images to generate")
    use_augmentation: bool = Field(True, description="Whether to use image augmentation")
    
class InpaintingBatchRequestModel(BaseModel):
    """
    Model representing a batch request for inpainting inference.
    """
    requests: List[InpaintingRequest]

async def save_image(image: UploadFile) -> str:
    """
    Save an uploaded image to a temporary file and return the file path.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        str: File path where the image is saved.
    """
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join("/tmp", file_name)
    with open(file_path, "wb") as f:
        f.write(await image.read())
    return file_path

def augment_image(image_path, target_width, target_height, roi_scale, segmentation_model_name, detection_model_name):
    """
    Augment an image by extending its dimensions and generating masks.

    Args:
        image_path (str): Path to the image file.
        target_width (int): Target width for augmentation.
        target_height (int): Target height for augmentation.
        roi_scale (float): Scale factor for region of interest.
        segmentation_model_name (str): Name of the segmentation model.
        detection_model_name (str): Name of the detection model.

    Returns:
        Tuple[Image.Image, Image.Image]: Augmented image and inverted mask.
    """
    image = Image.open(image_path)
=======
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
>>>>>>> 85a7460 (added API folder)
    image_augmentation = ImageAugmentation(target_width, target_height, roi_scale)
    image = image_augmentation.extend_image(image)
    mask = image_augmentation.generate_mask_from_bbox(image, segmentation_model_name, detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask

<<<<<<< HEAD
def run_inference(cfg, image_path: str, request: InpaintingRequest):
    """
    Run inference using an inpainting pipeline on an image.

    Args:
        cfg (dict): Configuration dictionary.
        image_path (str): Path to the image file.
        request (InpaintingRequest): Pydantic model containing inference parameters.

    Returns:
        dict: Resulting image in the specified mode ('b64_json' or 's3_json').
    
    Raises:
        ValueError: If an invalid mode is provided.
    """
    if request.use_augmentation:
        image, mask_image = augment_image(image_path, 
                                          cfg['target_width'], 
                                          cfg['target_height'], 
                                          cfg['roi_scale'], 
                                          cfg['segmentation_model'], 
                                          cfg['detection_model'])
    else:
        image = Image.open(image_path)
        mask_image = None  
    
    painting_pipeline = AutoPaintingPipeline(
        pipeline=inpainting_pipeline,
        image=image, 
        mask_image=mask_image, 
        target_height=cfg['target_height'], 
        target_width=cfg['target_width']
    )
    output = painting_pipeline.run_inference(prompt=request.prompt, 
                                    negative_prompt=request.negative_prompt, 
                                    num_inference_steps=request.num_inference_steps, 
                                    strength=request.strength, 
                                    guidance_scale=request.guidance_scale,
                                    num_images=request.num_images)
    if request.mode == "s3_json":
        return pil_to_s3_json(output, file_name="output.png")
    elif request.mode == "b64_json":
        return pil_to_b64_json(output)
    else:
        raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

class InpaintingBatcher(AsyncBatcher):
    async def process_batch(self, batch: Tuple[List[str], List[InpaintingRequest]]) -> List[Dict[str, Any]]:
        """
        Process a batch of images and requests for inpainting inference.

        Args:
            batch (Tuple[List[str], List[InpaintingRequest]]): Tuple of image paths and corresponding requests.

        Returns:
            List[Dict[str, Any]]: List of resulting images in the specified mode ('b64_json' or 's3_json').
        """
        image_paths, requests = batch
        results = []
        for image_path, request in zip(image_paths, requests):
            result = run_inference(cfg, image_path, request)
            results.append(result)
        return results

@router.post("/inpainting")
async def inpainting_inference(
    image: UploadFile = File(...),
    request_data: str = Form(...),
):
    """
    Handle POST request for inpainting inference.

    Args:
        image (UploadFile): Uploaded image file.
        request_data (str): JSON string of the request parameters.

    Returns:
        dict: Resulting image in the specified mode ('b64_json' or 's3_json').

    Raises:
        HTTPException: If there is an error during image processing.
    """
    try:
        image_path = await save_image(image)
        request_dict = json.loads(request_data)
        request = InpaintingRequest(**request_dict)
        result = run_inference(cfg, image_path, request)
=======
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
        
>>>>>>> 85a7460 (added API folder)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

<<<<<<< HEAD
@router.post("/inpainting/batch")
async def inpainting_batch_inference(
    images: List[UploadFile] = File(...),
    request_data: str = Form(...),
):
    """
    Handle POST request for batch inpainting inference.

    Args:
        images (List[UploadFile]): List of uploaded image files.
        request_data (str): JSON string of the request parameters.

    Returns:
        List[dict]: List of resulting images in the specified mode ('b64_json' or 's3_json').

    Raises:
        HTTPException: If there is an error during image processing.
    """
    try:
        request_dict = json.loads(request_data)
        batch_request = InpaintingBatchRequestModel(**request_dict)
        requests = batch_request.requests

        if len(images) != len(requests):
            raise HTTPException(status_code=400, detail="The number of images and requests must match.")

        batcher = InpaintingBatcher(max_batch_size=64)
        image_paths = [await save_image(image) for image in images]
        results = batcher.process_batch((image_paths, requests))

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
=======

>>>>>>> 85a7460 (added API folder)
