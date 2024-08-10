"""
SDXL Lora Inference API Client

This module provides a FastAPI-based client for interacting with an SDXL Lora Inference
Triton server. It offers image generation capabilities through a RESTful API.
"""

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pytriton.client import AsyncioModelClient

# Constants
TRITON_SERVER_URL = "localhost:8000"
TRITON_MODEL_NAME = "SDXL_Lora_Inference"
API_VERSION = "1.0.0"
DEFAULT_PORT = 8080

# Global variables
triton_client: Optional[AsyncioModelClient] = None


class ImageGenerationRequest(BaseModel):
    """
    Represents a request for image generation using the SDXL Lora model.
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
    Lifespan context manager for the FastAPI application.
    Handles initialization and cleanup of the Triton client.
    """
    global triton_client
    triton_client = AsyncioModelClient(TRITON_SERVER_URL, TRITON_MODEL_NAME)
    await triton_client.model_config
    print("Triton client initialized")
    yield
    if triton_client:
        await triton_client.close()
    print("Triton client closed")


app = FastAPI(
    title="SDXL Lora Inference API",
    description="API for generating images using SDXL Lora model",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v2/picpilot/docs',

)


@app.get("/api/v2/picpilot/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> Any:
    """
    Serve custom Swagger UI HTML with dark mode theme.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/gh/Itz-fork/Fastapi-Swagger-UI-Dark/assets/swagger_ui_dark.min.css",
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
        async with AsyncioModelClient.from_existing_client(
            triton_client
        ) as request_client:
            result_dict = await request_client.infer_sample(**inputs)

        output = result_dict["output"][0]
        return json.loads(output.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)
