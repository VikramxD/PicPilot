from fastapi import FastAPI, UploadFile, File,APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from utils import (accelerator, ImageAugmentation, clear_memory)
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import io

# Define FastAPI app
router = APIRouter()

class InpaintingRequest(BaseModel):
    
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int
    strength: float
    guidance_scale: float
    target_width: int
    target_height: int

class InpaintingBatchRequest(BaseModel):
    batch_input: List[InpaintingRequest]




def pil_to_s3_json(image: Image.Image, file_name: str):
    image_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(image_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(unique_file_name, exp=43200)  # 12 hours
    return {"image_id": image_id, "url": signed_url}

class AutoPaintingPipeline:
    def __init__(self, model_name: str, image: Image.Image, mask_image: Image.Image, target_width: int, target_height: int):
        self.model_name = model_name
        self.device = accelerator()
        self.pipeline = AutoPipelineForInpainting.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.image = load_image(image)
        self.mask_image = load_image(mask_image)
        self.target_width = target_width
        self.target_height = target_height
        self.pipeline.to(self.device)
        
    def run_inference(self, prompt: str, negative_prompt: Optional[str], num_inference_steps: int, strength: float, guidance_scale: float):
        clear_memory()
        image = load_image(self.image)
        mask_image = load_image(self.mask_image)
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            height=self.target_height,
            width=self.target_width
        ).images[0]
        return output

@app.post("/inpaint/")
async def inpaint(
    file: UploadFile = File(...),
    request: InpaintingRequest
):
    image = Image.open(file.file)
    augmenter = ImageAugmentation(target_width=request.target_width, target_height=request.target_height)  # Use fixed size or set dynamically
    extended_image = augmenter.extend_image(image)
    mask_image = augmenter.generate_mask_from_bbox(extended_image, 'segmentation_model', 'detection_model')
    mask_image = augmenter.invert_mask(mask_image)

    pipeline = AutoPaintingPipeline(
        model_name="model_name",
        image=extended_image,
        mask_image=mask_image,
        target_width=request.target_width,
        target_height=request.target_height
    )
    output_image = pipeline.run_inference(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        strength=request.strength,
        guidance_scale=request.guidance_scale,
        
    )

    
    result = pil_to_s3_json(output_image, "output_image.png")
    return result
