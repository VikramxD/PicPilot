import json
import io
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from pytriton.client import AsyncioModelClient
import logging
from PIL import Image
from scripts.s3_manager import S3ManagerService
from config_settings import settings

# Constants
TRITON_SERVER_URL = "0.0.0.0:8000"
TRITON_MODEL_NAME = "FLUX_INPAINTING_SERVER"
API_VERSION = "1.1.0"
DEFAULT_PORT = 8080

triton_client: Optional[AsyncioModelClient] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InpaintingRequest(BaseModel):
    """
    Represents a request for image inpainting.

    Attributes:
        prompt (str): The main prompt for inpainting.
        seed (int): Random seed for inpainting.
        strength (float): Strength of inpainting effect.
    """
    prompt: str = Field(..., description="The main prompt for inpainting")
    seed: int = Field(..., description="Random seed for inpainting")
    strength: float = Field(..., description="Strength of inpainting effect", ge=0.0, le=1.0)

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

def pil_to_s3_json(image: Image.Image, file_name) -> dict:
    """
    Uploads a PIL image to Amazon S3 and returns a JSON object containing the image ID, signed URL, and file name.

    Args:
        image (PIL.Image.Image): The PIL image to be uploaded.
        file_name (str): The name of the file.

    Returns:
        dict: A JSON object containing the image ID, signed URL, and file name.
    """
    image_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(image_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(unique_file_name, exp=43200)  # 12 hours
    return {"image_id": image_id, "url": signed_url, "file_name": unique_file_name}

def _prepare_inference_inputs(
    request: InpaintingRequest, image_data: dict, mask_data: dict
) -> Dict[str, np.ndarray]:
    """
    Prepare the inputs for the Triton inference server based on the request.

    Args:
        request (InpaintingRequest): The inpainting request.
        image_data (dict): Dictionary containing image URL and filename.
        mask_data (dict): Dictionary containing mask URL and filename.

    Returns:
        Dict[str, np.ndarray]: A dictionary of numpy arrays ready for inference.
    """
    return {
        "prompt": np.array([request.prompt.encode()], dtype=np.object_),
        "image_url": np.array([image_data['url'].encode()], dtype=np.object_),
        "image_filename": np.array([image_data['file_name'].encode()], dtype=np.object_),
        "mask_url": np.array([mask_data['url'].encode()], dtype=np.object_),
        "mask_filename": np.array([mask_data['file_name'].encode()], dtype=np.object_),
        "seed": np.array([request.seed], dtype=np.int32),
        "strength": np.array([request.strength], dtype=np.float32),
    }

app = FastAPI(
    title="PICPILOT API Suite",
    description="PICPILOT API CLIENT",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/api/v2/picpilot/docs',
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

@app.post("/upload_and_inpaint", response_model=Dict[str, Any])
async def upload_and_inpaint(
    prompt: str = Form(...),
    seed: int = Form(...),
    strength: float = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Upload image and mask files, convert to S3 URLs, and send them for inpainting.

    Args:
        prompt (str): The prompt for inpainting.
        seed (int): Random seed for inpainting.
        strength (float): Strength of inpainting effect.
        image (UploadFile): The image file to be inpainted.
        mask (UploadFile): The mask file to guide inpainting.

    Returns:
        Dict[str, Any]: A dictionary containing the generated inpainting result.
    """
    if not triton_client:
        raise HTTPException(status_code=500, detail="Triton client not initialized")

    try:
        # Convert UploadFile to PIL Image
        image_pil = Image.open(io.BytesIO(await image.read()))
        mask_pil = Image.open(io.BytesIO(await mask.read()))

        # Upload to S3 and get signed URLs and filenames
        image_data = pil_to_s3_json(image_pil, "image.png")
        mask_data = pil_to_s3_json(mask_pil, "mask.png")

        # Prepare inference inputs
        request = InpaintingRequest(prompt=prompt, seed=seed, strength=strength)
        inputs = _prepare_inference_inputs(request, image_data, mask_data)

        # Perform inference
        result_dict = await triton_client.infer_sample(**inputs)

        if "error" in result_dict:
            error_message = result_dict["error"][0].decode()
            raise HTTPException(status_code=400, detail=error_message)

        output = result_dict["output"][0]
        # Decode the JSON string and parse it
        output_data = json.loads(output.decode("utf-8"))
        
        # Convert the list back to a numpy array if needed
        output_array = np.array(output_data)

        # Process the output_array as needed (e.g., convert to image, upload to S3, etc.)
        # For now, we'll just return the data as-is
        return {"result": output_data}
    except Exception as e:
        logger.error(f"Error during inpainting: {e}")
        raise HTTPException(status_code=500, detail=f"Error during inpainting: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="debug")