import io
import base64
import time
import runpod
import asyncio
from typing import Dict, Any, AsyncGenerator
from PIL import Image
from pydantic import BaseModel, Field
from scripts.outpainting import Outpainter
from scripts.api_utils import pil_to_s3_json

# Global cache for the Outpainter instance
global_outpainter = None

class OutpaintingRequest(BaseModel):
    """
    Pydantic model representing a request for outpainting inference.
    """
    image: str = Field(..., description="Base64 encoded input image")
    width: int = Field(1024, description="Target width")
    height: int = Field(1024, description="Target height")
    overlap_percentage: int = Field(10, description="Mask overlap percentage")
    num_inference_steps: int = Field(8, description="Number of inference steps")
    resize_option: str = Field("Full", description="Resize option")
    custom_resize_percentage: int = Field(100, description="Custom resize percentage")
    prompt_input: str = Field("", description="Prompt for generation")
    alignment: str = Field("Middle", description="Image alignment")
    overlap_left: bool = Field(True, description="Apply overlap on left side")
    overlap_right: bool = Field(True, description="Apply overlap on right side")
    overlap_top: bool = Field(True, description="Apply overlap on top side")
    overlap_bottom: bool = Field(True, description="Apply overlap on bottom side")

async def initialize_model():
    """Initialize the model if not already loaded"""
    global global_outpainter
    if global_outpainter is None:
        print("Initializing Outpainter model...")
        global_outpainter = Outpainter()
        print("Model initialized successfully")
    return global_outpainter

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod.
    Yields status updates and progress during the outpainting process.
    """
    try:
        # Initial status
        yield {"status": "starting", "message": "Initializing outpainting process"}

        # Initialize model
        outpainter = await initialize_model()
        yield {"status": "processing", "message": "Model loaded successfully"}

        # Decode request
        try:
            request = OutpaintingRequest(**job['input'])
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            yield {"status": "processing", "message": "Request decoded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Start timing
        start_time = time.time()

        # Prepare outpainting parameters
        inputs = {
            'image': image,
            'params': {
                'width': request.width,
                'height': request.height,
                'overlap_percentage': request.overlap_percentage,
                'num_inference_steps': request.num_inference_steps,
                'resize_option': request.resize_option,
                'custom_resize_percentage': request.custom_resize_percentage,
                'prompt_input': request.prompt_input,
                'alignment': request.alignment,
                'overlap_left': request.overlap_left,
                'overlap_right': request.overlap_right,
                'overlap_top': request.overlap_top,
                'overlap_bottom': request.overlap_bottom
            }
        }

        yield {"status": "processing", "message": "Starting outpainting process"}

        # Perform outpainting
        try:
            result = outpainter.outpaint(
                inputs['image'],
                inputs['params']['width'],
                inputs['params']['height'],
                inputs['params']['overlap_percentage'],
                inputs['params']['num_inference_steps'],
                inputs['params']['resize_option'],
                inputs['params']['custom_resize_percentage'],
                inputs['params']['prompt_input'],
                inputs['params']['alignment'],
                inputs['params']['overlap_left'],
                inputs['params']['overlap_right'],
                inputs['params']['overlap_top'],
                inputs['params']['overlap_bottom']
            )
            yield {"status": "processing", "message": "Outpainting completed successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error during outpainting: {str(e)}"}
            return

        # Upload to S3 and get URL
        try:
            img_str = pil_to_s3_json(result, "outpainting_image")
            yield {"status": "processing", "message": "Image uploaded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error uploading image: {str(e)}"}
            return

        # Calculate completion time
        completion_time = time.time() - start_time

        # Final response
        final_response = {
            "status": "completed",
            "output": {
                "result": img_str,
                "completion_time": round(completion_time, 2),
                "image_resolution": f"{result.width}x{result.height}"
            }
        }

        yield final_response

    except Exception as e:
        yield {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

# Initialize the model when the service starts
print("Initializing service...")
asyncio.get_event_loop().run_until_complete(initialize_model())
print("Service initialization complete")

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
    })