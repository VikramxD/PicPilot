import base64
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field, ValidationError
from pytriton.client import AsyncioModelClient
import logging
from PIL import Image
import io

# Constants
TRITON_SERVER_URL = "0.0.0.0:8000"
TRITON_MODEL_NAME = "outpainting"
API_VERSION = "1.0.0"
DEFAULT_PORT = 8080

triton_client: Optional[AsyncioModelClient] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutpaintingRequest(BaseModel):
    """
    Represents a request for outpainting using the ControlNetZoeDepthOutpainting model.
    """
    base_image_description: str = Field(..., description="Description of the base image")
    base_image_negative_prompt: str = Field("", description="Negative prompt for the base image")
    controlnet_conditioning_scale: float = Field(1.0, description="ControlNet conditioning scale", ge=0.0, le=2.0)
    controlnet_guidance_scale: float = Field(7.5, description="ControlNet guidance scale", ge=0.0, le=20.0)
    controlnet_num_inference_steps: int = Field(50, description="Number of inference steps for ControlNet", ge=1, le=150)
    controlnet_guidance_end: float = Field(1.0, description="ControlNet guidance end", ge=0.0, le=1.0)
    background_extension_prompt: str = Field("", description="Prompt for background extension")
    outpainting_negative_prompt: str = Field("", description="Negative prompt for outpainting")
    outpainting_guidance_scale: float = Field(7.5, description="Guidance scale for outpainting", ge=0.0, le=20.0)
    outpainting_strength: float = Field(0.5, description="Strength of outpainting", ge=0.0, le=1.0)
    outpainting_num_inference_steps: int = Field(50, description="Number of inference steps for outpainting", ge=1, le=150)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager for the FastAPI application.
    Handles initialization and cleanup of the Triton client.
    """
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

def _prepare_inference_inputs(image: np.ndarray, request: OutpaintingRequest) -> Dict[str, np.ndarray]:
    """
    Prepare the inputs for the Triton inference server based on the request and image.
    """
    return {
        "image": image,
        "base_image_description": np.array([[request.base_image_description.encode()]], dtype=np.object_),
        "base_image_negative_prompt": np.array([[request.base_image_negative_prompt.encode()]], dtype=np.object_),
        "controlnet_conditioning_scale": np.array([[request.controlnet_conditioning_scale]], dtype=np.float32),
        "controlnet_guidance_scale": np.array([[request.controlnet_guidance_scale]], dtype=np.float32),
        "controlnet_num_inference_steps": np.array([[request.controlnet_num_inference_steps]], dtype=np.int32),
        "controlnet_guidance_end": np.array([[request.controlnet_guidance_end]], dtype=np.float32),
        "background_extension_prompt": np.array([[request.background_extension_prompt.encode()]], dtype=np.object_),
        "outpainting_negative_prompt": np.array([[request.outpainting_negative_prompt.encode()]], dtype=np.object_),
        "outpainting_guidance_scale": np.array([[request.outpainting_guidance_scale]], dtype=np.float32),
        "outpainting_strength": np.array([[request.outpainting_strength]], dtype=np.float32),
        "outpainting_num_inference_steps": np.array([[request.outpainting_num_inference_steps]], dtype=np.int32),
    }

app = FastAPI(
    title="Outpainting API",
    description="Outpainting API using ControlNetZoeDepthOutpainting",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v1/outpainting/docs',
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

@app.post("/outpaint", response_model=Dict[str, Any])
async def outpaint(
    image: UploadFile = File(...),
    request: str = Form(...)
) -> Dict[str, Any]:
    """
    Perform outpainting on the provided image based on the given parameters.
    """
    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized")

    try:
        # Parse the request JSON string
        request_data = json.loads(request)
        outpainting_request = OutpaintingRequest(**request_data)

        # Read and preprocess the image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        np_image = np.array(pil_image)

        # Prepare inputs
        inputs = _prepare_inference_inputs(np_image, outpainting_request)

        # Perform inference
        result_dict = await triton_client.infer_sample(**inputs)

        # Process the output
        output = result_dict["result"][0]
        result_json = json.loads(output.decode("utf-8"))

        # If the result is a base64 encoded image, decode it
        if "image" in result_json and isinstance(result_json["image"], str):
            image_data = base64.b64decode(result_json["image"])
            result_json["image"] = base64.b64encode(image_data).decode("utf-8")

        return result_json
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request data")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error performing outpainting: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing outpainting: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="debug")
