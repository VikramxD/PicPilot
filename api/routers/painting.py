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
import asyncio
pl.seed_everything(42)
router = APIRouter()

# Initialize Hydra configuration
with initialize(version_base=None, config_path="../../configs"):
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

def run_inference(cfg, image_path: str, mask_image_path: str, request: InpaintingRequest):
    """
    Run inference using an inpainting pipeline on an image.

    Args:
        cfg (dict): Configuration dictionary.
        image_path (str): Path to the image file.
        mask_image_path (str): Path to the mask image file.
        request (InpaintingRequest): Pydantic model containing inference parameters.

    Returns:
        dict: Resulting image in the specified mode ('b64_json' or 's3_json').
    
    Raises:
        ValueError: If an invalid mode is provided.
    """
    image = Image.open(image_path)
    mask_image = Image.open(mask_image_path)
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
    def __init__(self, max_batch_size: int):
        super().__init__(max_batch_size)

    async def process_batch(self, batch: Tuple[List[str], List[str], List[InpaintingRequest]]) -> List[Dict[str, Any]]:
        """
        Process a batch of images and requests for inpainting inference.

        Args:
            batch (Tuple[List[str], List[str], List[InpaintingRequest]]): Tuple of image paths, mask image paths, and corresponding requests.

        Returns:
            List[Dict[str, Any]]: List of resulting images in the specified mode ('b64_json' or 's3_json').
        """
        image_paths, mask_image_paths, requests = batch
        results = []
        for image_path, mask_image_path, request in zip(image_paths, mask_image_paths, requests):
            try:
                result = run_inference(cfg, image_path, mask_image_path, request)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results

@router.post("/inpainting")
async def inpainting_inference(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    request_data: str = Form(...),
):
    """
    Handle POST request for inpainting inference.

    Args:
        image (UploadFile): Uploaded image file.
        mask_image (UploadFile): Uploaded mask image file.
        request_data (str): JSON string of the request parameters.

    Returns:
        dict: Resulting image in the specified mode ('b64_json' or 's3_json').

    Raises:
        HTTPException: If there is an error during image processing.
    """
    try:
        image_path = await save_image(image)
        mask_image_path = await save_image(mask_image)
        request_dict = json.loads(request_data)
        request = InpaintingRequest(**request_dict)
        result = run_inference(cfg, image_path, mask_image_path, request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inpainting/batch")
async def inpainting_batch_inference(
    images: List[UploadFile] = File(...),
    mask_images: List[UploadFile] = File(...),
    request_data: str = Form(...),
):
    """
    Handle POST request for batch inpainting inference.

    Args:
        images (List[UploadFile]): List of uploaded image files.
        mask_images (List[UploadFile]): List of uploaded mask image files.
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

        if len(images) != len(requests) or len(images) != len(mask_images):
            raise HTTPException(status_code=400, detail="The number of images, mask images, and requests must match.")

        batcher = InpaintingBatcher(max_batch_size=64)
        image_paths = [save_image(image) for image in images]
        mask_image_paths = [save_image(mask_image) for mask_image in mask_images]
        results = await batcher.process_batch((image_paths, mask_image_paths, requests))

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))