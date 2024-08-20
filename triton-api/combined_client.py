import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from pytriton.client import AsyncioModelClient
import logging
from PIL import Image
import io
import base64

# Constants
TRITON_SERVER_URL = "localhost:8000"
TRITON_MODEL_NAME = "outpainting"
API_VERSION = "1.0.0"
DEFAULT_PORT = 8080

triton_client: Optional[AsyncioModelClient] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutpaintingRequest(BaseModel):
    controlnet_prompt: str
    controlnet_negative_prompt: str
    controlnet_conditioning_scale: float
    controlnet_guidance_scale: float
    controlnet_num_inference_steps: int
    controlnet_guidance_end: float
    inpainting_prompt: str
    inpainting_negative_prompt: str
    inpainting_guidance_scale: float
    inpainting_strength: float
    inpainting_num_inference_steps: int

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

def _prepare_inference_inputs(image: np.ndarray, request: OutpaintingRequest) -> Dict[str, np.ndarray]:
    return {
        "image": image.flatten(),
        "controlnet_prompt": np.array([request.controlnet_prompt.encode()], dtype=np.object_),
        "controlnet_negative_prompt": np.array([request.controlnet_negative_prompt.encode()], dtype=np.object_),
        "controlnet_conditioning_scale": np.array([request.controlnet_conditioning_scale], dtype=np.float32),
        "controlnet_guidance_scale": np.array([request.controlnet_guidance_scale], dtype=np.float32),
        "controlnet_num_inference_steps": np.array([request.controlnet_num_inference_steps], dtype=np.int32),
        "controlnet_guidance_end": np.array([request.controlnet_guidance_end], dtype=np.float32),
        "inpainting_prompt": np.array([request.inpainting_prompt.encode()], dtype=np.object_),
        "inpainting_negative_prompt": np.array([request.inpainting_negative_prompt.encode()], dtype=np.object_),
        "inpainting_guidance_scale": np.array([request.inpainting_guidance_scale], dtype=np.float32),
        "inpainting_strength": np.array([request.inpainting_strength], dtype=np.float32),
        "inpainting_num_inference_steps": np.array([request.inpainting_num_inference_steps], dtype=np.int32),
    }

app = FastAPI(
    title="ControlNetZoeDepthOutpainting API",
    description="API for ControlNetZoeDepthOutpainting using PyTriton",
    version=API_VERSION,
    lifespan=lifespan,
)

@app.post("/outpaint", response_model=Dict[str, Any])
async def outpaint(file: UploadFile = File(...), request: str = Form(...)):
    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized")

    try:
        # Parse the JSON string into a dictionary
        request_dict = json.loads(request)
        
        # Create an OutpaintingRequest object
        outpainting_request = OutpaintingRequest(**request_dict)
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)

        # Prepare inputs for the Triton server
        inputs = _prepare_inference_inputs(image_array, outpainting_request)

        # Send request to Triton server
        result = await triton_client.infer_sample(**inputs)

        # Process the result
        output_image = result["output_image"]
        output_pil = Image.fromarray(output_image.astype('uint8'))
        
        # Convert the result to base64
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_str}
    except Exception as e:
        logger.error(f"Error processing outpainting request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing outpainting request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)
