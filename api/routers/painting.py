from fastapi import  APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image
import sys
sys.path.append("../scripts")
import uuid
import lightning.pytorch as pl
from typing import List
from utils import pil_to_s3_json, pil_to_b64_json, ImageAugmentation, accelerator
from inpainting_pipeline import AutoPaintingPipeline, load_pipeline
from hydra import compose, initialize
from pydantic import BaseModel
from async_batcher.batcher import AsyncBatcher
from typing import Dict


router = APIRouter()
pl.seed_everything(42)

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="inpainting")
inpainting_pipeline = load_pipeline(cfg.model, accelerator(), enable_compile=True)

def augment_image(image_path, target_width, target_height, roi_scale, segmentation_model_name, detection_model_name):
    image = Image.open(image_path)
    image_augmentation = ImageAugmentation(target_width, target_height, roi_scale)
    image = image_augmentation.extend_image(image)
    mask = image_augmentation.generate_mask_from_bbox(image, segmentation_model_name, detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask

def run_inference(cfg: dict, image_path: str, prompt: str, negative_prompt: str, num_inference_steps: int, strength: float, guidance_scale: float, mode: str, num_images: int):
    image, mask_image = augment_image(image_path, 
                                      cfg['target_width'], 
                                      cfg['target_height'], 
                                      cfg['roi_scale'], 
                                      cfg['segmentation_model'], 
                                      cfg['detection_model'])
    
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

class InpaintingRequest(BaseModel):
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    strength: float
    guidance_scale: float
    num_images: int = 1

class InpaintingBatcher(AsyncBatcher[List[Dict], dict]):
    def __init__(self, pipeline, cfg):
        self.pipeline = pipeline
        self.cfg = cfg

    def process_batch(self, batch: List[Dict], image_paths: List[str]) -> List[dict]:
        results = []
        for data, image_path in zip(batch, image_paths):
            try:
                image, mask_image = augment_image(
                    image_path,
                    self.cfg['target_width'],
                    self.cfg['target_height'],
                    self.cfg['roi_scale'],
                    self.cfg['segmentation_model'],
                    self.cfg['detection_model']
                )
                
                pipeline = AutoPaintingPipeline(
                    image=image, 
                    mask_image=mask_image, 
                    target_height=self.cfg['target_height'], 
                    target_width=self.cfg['target_width']
                )
                output = pipeline.run_inference(
                    prompt=data['prompt'], 
                    negative_prompt=data['negative_prompt'], 
                    num_inference_steps=data['num_inference_steps'], 
                    strength=data['strength'], 
                    guidance_scale=data['guidance_scale']
                )

                if data['mode'] == "s3_json":
                    result = pil_to_s3_json(output, 'inpainting_image')
                elif data['mode'] == "b64_json":
                    result = pil_to_b64_json(output)
                else:
                    raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")
                
                results.append(result)
            except Exception as e:
                print(f"Error in process_batch: {e}")
                raise HTTPException(status_code=500, detail="Batch inference failed")
        return results

@router.post("/inpainting")
async def inpainting_inference(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    num_inference_steps: int = Form(...),
    strength: float = Form(...),
    guidance_scale: float = Form(...),
    mode: str = Form(...),
    num_images: int = Form(1)
):
    """
    Run the inpainting/outpainting inference pipeline.

    Parameters:
    - image: UploadFile - The image file to be used for inpainting/outpainting.
    - prompt: str - The prompt text for guiding the inpainting/outpainting process.
    - negative_prompt: str - The negative prompt text for guiding the inpainting/outpainting process.
    - num_inference_steps: int - The number of inference steps to perform during the inpainting/outpainting process.
    - strength: float - The strength parameter for controlling the inpainting/outpainting process.
    - guidance_scale: float - The guidance scale parameter for controlling the inpainting/outpainting process.
    - mode: str - The output mode, either "s3_json" or "b64_json".
    - num_images: int - The number of images to generate.

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
        
        result = run_inference(
            cfg, image_path, prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inpainting_batch")
async def inpainting_batch_inference(
    batch: List[dict], 
    images: List[UploadFile] = File(...)
):
    """
    Run batch inpainting/outpainting inference pipeline.

    Parameters:
    - batch: List[dict] - The batch of requests containing parameters for the inpainting/outpainting process.
    - images: List[UploadFile] - The list of image files to be used for inpainting/outpainting.

    Returns:
    - results: The results of the inpainting/outpainting process for each request.

    Raises:
    - HTTPException: If an error occurs during the inpainting/outpainting process.
    """
    try:
        image_paths = []
        for image in images:
            image_bytes = await image.read()
            image_path = f"/tmp/{uuid.uuid4()}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)

        batcher = InpaintingBatcher(pipeline, cfg)
        results = batcher.process_batch(batch, image_paths)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


