import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pytriton.client import AsyncioModelClient
import logging

# Constants
TRITON_SERVER_URL = "0.0.0.0:8000"
TRITON_MODEL_NAME = "TTI_SDXL_KREAM"
API_VERSION = "1.1.0"
DEFAULT_PORT = 8080

# Global variables
triton_client: Optional[AsyncioModelClient] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerationRequest(BaseModel):
    """
    Represents a request for image generation using the SDXL Lora model.

    Attributes:
        prompt (str): The main prompt for image generation.
        negative_prompt (str): Negative prompt to guide what to avoid in the image.
        num_images (int): Number of images to generate.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        mode (str): Output mode: 'b64_json' or 's3_json'.
    """

    prompt: str = Field(..., description="The main prompt for image generation")
    negative_prompt: str = Field(
        "", description="Negative prompt to guide what to avoid in the image"
    )
    num_images: int = Field(1, description="Number of images to generate", ge=1, le=10)
    num_inference_steps: int = Field(
        50, description="Number of denoising steps", ge=1, le=150
    )
    guidance_scale: float = Field(
        7.5, description="Scale for classifier-free guidance", ge=1.0, le=20.0
    )
    mode: str = Field("b64_json", description="Output mode: 'b64_json' or 's3_json'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager for the FastAPI application.
    Handles initialization and cleanup of the Triton client.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
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


def _prepare_inference_inputs(request: ImageGenerationRequest) -> Dict[str, np.ndarray]:
    """
    Prepare the inputs for the Triton inference server based on the request.

    Args:
        request (ImageGenerationRequest): The image generation request.

    Returns:
        Dict[str, np.ndarray]: A dictionary of numpy arrays ready for inference.
    """
    return {
        "prompt": np.array([request.prompt.encode()], dtype=np.object_),
        "negative_prompt": np.array(
            [request.negative_prompt.encode()], dtype=np.object_
        ),
        "num_images": np.array([request.num_images], dtype=np.int32),
        "num_inference_steps": np.array([request.num_inference_steps], dtype=np.int32),
        "guidance_scale": np.array([request.guidance_scale], dtype=np.float32),
        "mode": np.array([request.mode.encode()], dtype=np.object_),
    }


app = FastAPI(
    title="PICPILOT API Suite",
    description="API for generating images using SDXL Lora model",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v2/picpilot/docs',
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)


@app.post("/generate_image", response_model=Dict[str, Any])
async def generate_image(request: ImageGenerationRequest) -> Dict[str, Any]:
    """
    Generate an image based on the provided parameters using the SDXL Lora model.

    Args:
        request (ImageGenerationRequest): The parameters for image generation.

    Returns:
        Dict[str, Any]: A dictionary containing the generated image data or URL.

    Raises:
        HTTPException: If the Triton client is not initialized or if there's an error in image generation.
    """
    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized")

    try:
        inputs = _prepare_inference_inputs(request)
        result_dict = await triton_client.infer_sample(**inputs)

        output = result_dict["output"][0]
        return json.loads(output.decode("utf-8"))
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="debug")
