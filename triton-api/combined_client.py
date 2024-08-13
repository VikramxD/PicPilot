import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from pytriton.client import AsyncioModelClient
import logging

# Constants
TRITON_SERVER_URL = "0.0.0.0:8000"
SDXL_MODEL_NAME = "sdxl_lora"
OUTPAINTING_MODEL_NAME = "outpainting"
API_VERSION = "1.1.0"
DEFAULT_PORT = 8080

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The main prompt for image generation")
    negative_prompt: str = Field("", description="Negative prompt to guide what to avoid in the image")
    num_images: int = Field(1, description="Number of images to generate", ge=1, le=10)
    num_inference_steps: int = Field(50, description="Number of denoising steps", ge=1, le=150)
    guidance_scale: float = Field(7.5, description="Scale for classifier-free guidance", ge=1.0, le=20.0)
    mode: str = Field("b64_json", description="Output mode: 'b64_json' or 's3_json'")

class OutpaintingRequest(BaseModel):
    controlnet_prompt: str = Field(..., description="The prompt for ControlNet")
    controlnet_negative_prompt: str = Field("", description="Negative prompt for ControlNet")
    controlnet_conditioning_scale: float = Field(0.9, description="Conditioning scale for ControlNet")
    controlnet_guidance_scale: float = Field(7.5, description="Guidance scale for ControlNet")
    controlnet_num_inference_steps: int = Field(50, description="Number of inference steps for ControlNet")
    controlnet_guidance_end: float = Field(0.9, description="Guidance end for ControlNet")
    inpainting_prompt: str = Field(..., description="The prompt for inpainting")
    inpainting_negative_prompt: str = Field("", description="Negative prompt for inpainting")
    inpainting_guidance_scale: float = Field(7.5, description="Guidance scale for inpainting")
    inpainting_strength: float = Field(0.8, description="Strength for inpainting")
    inpainting_num_inference_steps: int = Field(50, description="Number of inference steps for inpainting")
    mode: str = Field("b64_json", description="Output mode: 'b64_json' or 's3_json'")

sdxl_client: Optional[AsyncioModelClient] = None
outpainting_client: Optional[AsyncioModelClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sdxl_client, outpainting_client
    try:
        sdxl_client = AsyncioModelClient(TRITON_SERVER_URL, SDXL_MODEL_NAME)
        outpainting_client = AsyncioModelClient(TRITON_SERVER_URL, OUTPAINTING_MODEL_NAME)
        await sdxl_client.model_config
        await outpainting_client.model_config
        logger.info("Triton clients initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize Triton clients: {e}")
        raise HTTPException(status_code=500, detail="Error initializing Triton clients")
    finally:
        if sdxl_client:
            await sdxl_client.close()
        if outpainting_client:
            await outpainting_client.close()
        logger.info("Triton clients closed")

def _prepare_sdxl_inputs(request: ImageGenerationRequest) -> Dict[str, np.ndarray]:
    return {
        "prompt": np.array([request.prompt.encode()], dtype=np.object_),
        "negative_prompt": np.array([request.negative_prompt.encode()], dtype=np.object_),
        "num_images": np.array([request.num_images], dtype=np.int32),
        "num_inference_steps": np.array([request.num_inference_steps], dtype=np.int32),
        "guidance_scale": np.array([request.guidance_scale], dtype=np.float32),
        "mode": np.array([request.mode.encode()], dtype=np.object_),
    }

def _prepare_outpainting_inputs(image: bytes, request: OutpaintingRequest) -> Dict[str, np.ndarray]:
    return {
        "image": np.array([image], dtype=np.object_),
        "request": np.array([json.dumps(request.dict()).encode()], dtype=np.object_),
    }

app = FastAPI(
    title="PICPILOT API Suite",
    description="PICPILOT API CLIENT",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v2/picpilot/docs',
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

@app.post("/generate_image", response_model=Dict[str, Any])
async def generate_image(request: ImageGenerationRequest) -> Dict[str, Any]:
    if not sdxl_client:
        raise HTTPException(status_code=500, detail="SDXL client not initialized")

    try:
        inputs = _prepare_sdxl_inputs(request)
        result_dict = await sdxl_client.infer_sample(**inputs)
        output = result_dict["output"][0]
        return json.loads(output.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/outpaint_image", response_model=Dict[str, Any])
async def outpaint_image(image: UploadFile = File(...), request: OutpaintingRequest = None) -> Dict[str, Any]:
    if not outpainting_client:
        raise HTTPException(status_code=500, detail="Outpainting client not initialized")

    try:
        image_bytes = await image.read()
        inputs = _prepare_outpainting_inputs(image_bytes, request)
        result_dict = await outpainting_client.infer_sample(**inputs)
        output = result_dict["output"][0]
        return json.loads(output.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error outpainting image: {e}")
        raise HTTPException(status_code=500, detail=f"Error outpainting image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="debug")