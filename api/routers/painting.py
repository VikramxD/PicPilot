import sys
sys.path.append('../scripts')
import os
import uuid
from typing import List, Tuple, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image
import lightning.pytorch as pl
from utils import pil_to_s3_json, pil_to_b64_json, ImageAugmentation, accelerator
from inpainting_pipeline import AutoPaintingPipeline, load_pipeline
from hydra import compose, initialize
from async_batcher.batcher import AsyncBatcher

pl.seed_everything(42)
router = APIRouter()

# Initialize the configuration and pipeline
with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="inpainting")
inpainting_pipeline = load_pipeline(cfg.model, accelerator(), enable_compile=True)

async def save_image(image: UploadFile) -> str:
    """Save an uploaded image to a temporary file and return the file path."""
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join("/tmp", file_name)
    with open(file_path, "wb") as f:
        f.write(await image.read())
    return file_path

def augment_image(image_path, target_width, target_height, roi_scale, segmentation_model_name, detection_model_name):
    image = Image.open(image_path)
    image_augmentation = ImageAugmentation(target_width, target_height, roi_scale)
    image = image_augmentation.extend_image(image)
    mask = image_augmentation.generate_mask_from_bbox(image, segmentation_model_name, detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask

def run_inference(cfg,
                  image_path: str, 
                  prompt: str, 
                  negative_prompt: str, 
                  num_inference_steps: int, 
                  strength: float, 
                  guidance_scale: float, 
                  mode: str, 
                  num_images: int,
                  use_augmentation: bool):
    if use_augmentation:
        image, mask_image = augment_image(image_path, 
                                          cfg['target_width'], 
                                          cfg['target_height'], 
                                          cfg['roi_scale'], 
                                          cfg['segmentation_model'], 
                                          cfg['detection_model'])
    else:
        image = Image.open(image_path)
        mask_image = None  # Assume mask_image is provided or generated separately
    
    painting_pipeline = AutoPaintingPipeline(
        pipeline=inpainting_pipeline,
        image=image, 
        mask_image=mask_image, 
        target_height=cfg['target_height'], 
        target_width=cfg['target_width']
    )
    output = painting_pipeline.run_inference(prompt=prompt, 
                                    negative_prompt=negative_prompt, 
                                    num_inference_steps=num_inference_steps, 
                                    strength=strength, 
                                    guidance_scale=guidance_scale)
    if mode == "s3_json":
        return pil_to_s3_json(output, file_name="output.png")
    elif mode == "b64_json":
        return pil_to_b64_json(output)
    else:
        raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")



@router.post("/inpainting")
async def inpainting_inference(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    num_inference_steps: int = Form(...),
    strength: float = Form(...),
    guidance_scale: float = Form(...),
    mode: str = Form(...),
    num_images: int = Form(1),
    
):
    try:
        image_path = await save_image(image)
        result = run_inference(cfg, image_path, prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images, use_augmentation=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


