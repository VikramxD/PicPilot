import os
import uuid
import json
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field, ValidationError
import lightning.pytorch as pl
from scripts.api_utils import pil_to_b64_json
from scripts.outpainting import ControlNetZoeDepthOutpainting
from async_batcher.batcher import AsyncBatcher
from functools import lru_cache

pl.seed_everything(42)
router = APIRouter()

@lru_cache(maxsize=1)
def load_pipeline():
    outpainting_pipeline = ControlNetZoeDepthOutpainting(target_size=(1024, 1024))
    return outpainting_pipeline

class OutpaintingRequest(BaseModel):
    """
    Model representing a request for outpainting inference.
    """
    controlnet_prompt: str = Field(...)
    controlnet_negative_prompt: str = Field(...)
    controlnet_conditioning_scale: float = Field(...)
    controlnet_guidance_scale: float = Field(...)
    controlnet_num_inference_steps: int = Field(...)
    controlnet_guidance_end: float = Field(...)
    inpainting_prompt: str = Field(...)
    inpainting_negative_prompt: str = Field(...)
    inpainting_guidance_scale: float = Field(...)
    inpainting_strength: float = Field(...)
    inpainting_num_inference_steps: int = Field(...)

class OutpaintingBatchRequestModel(BaseModel):
    """
    Model representing a batch request for outpainting inference.
    """
    requests: List[OutpaintingRequest]

async def save_image(image: UploadFile) -> str:
    """
    Save an uploaded image to a temporary file and return the file path.
    """
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join("/tmp", file_name)
    with open(file_path, "wb") as f:
        f.write(await image.read())
    return file_path

def run_inference(image_path: str, request: OutpaintingRequest):
    pipeline = load_pipeline()
    result = pipeline.run_pipeline(
        image_path,
        controlnet_prompt=request.controlnet_prompt,
        controlnet_negative_prompt=request.controlnet_negative_prompt,
        controlnet_conditioning_scale=request.controlnet_conditioning_scale,
        controlnet_guidance_scale=request.controlnet_guidance_scale,
        controlnet_num_inference_steps=request.controlnet_num_inference_steps,
        controlnet_guidance_end=request.controlnet_guidance_end,
        inpainting_prompt=request.inpainting_prompt,
        inpainting_negative_prompt=request.inpainting_negative_prompt,
        inpainting_guidance_scale=request.inpainting_guidance_scale,
        inpainting_strength=request.inpainting_strength,
        inpainting_num_inference_steps=request.inpainting_num_inference_steps
    )
    return result

@router.post("/outpaint")
async def outpaint(
    image: UploadFile = File(...),
    request: str = Form(...)
):
    try:
        request_dict = json.loads(request)
        outpainting_request = OutpaintingRequest(**request_dict)
        
        image_path = await save_image(image)
        result = run_inference(image_path, outpainting_request)
        
        result_json = pil_to_b64_json(result)
        
        os.remove(image_path)
        
        return {"result": result_json}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request data")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class OutpaintingBatcher(AsyncBatcher):
    async def process_batch(self, batch):
        results = []
        for image, request in batch:
            image_path = await save_image(image)
            try:
                result = run_inference(image_path, request)
                results.append(result)
            finally:
                os.remove(image_path)
        return results

@router.post("/batch_outpaint")
async def batch_outpaint(images: List[UploadFile] = File(...), batch_request: str = Form(...)):
    try:
        batch_request_dict = json.loads(batch_request)
        batch_outpainting_request = OutpaintingBatchRequestModel(**batch_request_dict)
        
        batcher = OutpaintingBatcher(max_queue_size=64)
        results = await batcher.process_batch(list(zip(images, batch_outpainting_request.requests)))
        
        result_jsons = [pil_to_b64_json(result) for result in results]

        return {"results": result_jsons}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in batch request data")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))