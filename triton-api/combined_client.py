import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from pytriton.client import AsyncioModelClient
import logging

# Constants
TRITON_SERVER_URL = "0.0.0.0:8000"
TRITON_MODEL_NAME = "outpainting"
API_VERSION = "1.0.0"
DEFAULT_PORT = 7860

triton_client: Optional[AsyncioModelClient] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutpaintingParams(BaseModel):
    base_image_description: str
    base_image_negative_prompt: str = ""
    controlnet_conditioning_scale: float = Field(1.0, ge=0.0, le=2.0)
    controlnet_guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    controlnet_num_inference_steps: int = Field(50, ge=1, le=150)
    controlnet_guidance_end: float = Field(1.0, ge=0.0, le=1.0)
    background_extension_prompt: str = ""
    outpainting_negative_prompt: str = ""
    outpainting_guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    outpainting_strength: float = Field(0.5, ge=0.0, le=1.0)
    outpainting_num_inference_steps: int = Field(50, ge=1, le=150)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global triton_client
    try:
        triton_client = AsyncioModelClient(TRITON_SERVER_URL, TRITON_MODEL_NAME)
        await triton_client.model_config
        logger.info("Triton client initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        raise HTTPException(status_code=500, detail="Error initializing Triton client")
    finally:
        if triton_client:
            await triton_client.close()
        logger.info("Triton client closed")

def _prepare_inference_inputs(image: bytes, params: OutpaintingParams) -> Dict[str, np.ndarray]:
    return {
        "image": np.array([image], dtype=np.bytes_),
        "base_image_description": np.array([[params.base_image_description.encode()]], dtype=np.bytes_),
        "base_image_negative_prompt": np.array([[params.base_image_negative_prompt.encode()]], dtype=np.bytes_),
        "controlnet_conditioning_scale": np.array([[params.controlnet_conditioning_scale]], dtype=np.float32),
        "controlnet_guidance_scale": np.array([[params.controlnet_guidance_scale]], dtype=np.float32),
        "controlnet_num_inference_steps": np.array([[params.controlnet_num_inference_steps]], dtype=np.int32),
        "controlnet_guidance_end": np.array([[params.controlnet_guidance_end]], dtype=np.float32),
        "background_extension_prompt": np.array([[params.background_extension_prompt.encode()]], dtype=np.bytes_),
        "outpainting_negative_prompt": np.array([[params.outpainting_negative_prompt.encode()]], dtype=np.bytes_),
        "outpainting_guidance_scale": np.array([[params.outpainting_guidance_scale]], dtype=np.float32),
        "outpainting_strength": np.array([[params.outpainting_strength]], dtype=np.float32),
        "outpainting_num_inference_steps": np.array([[params.outpainting_num_inference_steps]], dtype=np.int32),
    }

app = FastAPI(
    title="Outpainting API Client",
    description="Client for Outpainting API using ControlNetZoeDepthOutpainting",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v2/outpainting/docs',
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

@app.post("/outpaint", response_model=Dict[str, Any])
async def outpaint(
    image: UploadFile = File(...),
    params: str = Form(...)
) -> Dict[str, Any]:
    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized")

    try:
        # Parse the parameters
        outpainting_params = OutpaintingParams.parse_raw(params)
        
        # Read the image file
        image_bytes = await image.read()

        # Prepare inputs
        inputs = _prepare_inference_inputs(image_bytes, outpainting_params)

        # Perform inference
        result_dict = await triton_client.infer_sample(**inputs)

        # Process the output
        output = result_dict["result"][0]
        result_json = json.loads(output.decode("utf-8"))

        return result_json
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request data")
    except Exception as e:
        logger.error(f"Error performing outpainting: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing outpainting: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="debug")