import io
import base64
import time
import asyncio
from typing import Dict, Any, Tuple, AsyncGenerator
from PIL import Image
from pydantic import BaseModel, Field
from scripts.outpainting import Outpainter
from scripts.api_utils import pil_to_s3_json

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

async def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming request asynchronously.
    """
    try:
        outpainting_request = OutpaintingRequest(**request)
        image_data = await asyncio.to_thread(
            base64.b64decode, outpainting_request.image
        )
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data)).convert("RGBA")
        )
        
        return {
            'image': image,
            'params': outpainting_request.model_dump()
        }
    except Exception as e:
        raise ValueError(f"Invalid request: {str(e)}")

async def generate_outpainting(inputs: Dict[str, Any]) -> Tuple[Image.Image, float]:
    """
    Perform outpainting operation asynchronously.
    """
    start_time = time.time()
    
    # Initialize Outpainter for each request
    outpainter = Outpainter()
    
    result = await asyncio.to_thread(
        outpainter.outpaint,
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
    
    completion_time = time.time() - start_time
    return result, completion_time

async def format_response(result: Image.Image, completion_time: float) -> Dict[str, Any]:
    """
    Format the outpainting result for API response asynchronously.
    """
    img_str = await asyncio.to_thread(
        pil_to_s3_json,
        result,
        "outpainting_image"
    )
    
    return {
        "result": img_str,
        "completion_time": round(completion_time, 2),
        "image_resolution": f"{result.width}x{result.height}"
    }

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod with progress updates.
    """
    try:
        # Initial status
        yield {"status": "starting", "message": "Starting outpainting process"}

        # Decode request
        try:
            inputs = await decode_request(job['input'])
            yield {
                "status": "processing", 
                "message": "Request decoded successfully",
                "input_resolution": f"{inputs['image'].width}x{inputs['image'].height}"
            }
        except Exception as e:
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Generate outpainting
        try:
            yield {"status": "processing", "message": "Initializing outpainting model"}
            result, completion_time = await generate_outpainting(inputs)
            yield {
                "status": "processing",
                "message": "Outpainting completed",
                "completion_time": f"{completion_time:.2f}s"
            }
        except Exception as e:
            yield {"status": "error", "message": f"Error during outpainting: {str(e)}"}
            return

        # Format and upload result
        try:
            yield {"status": "processing", "message": "Uploading result"}
            response = await format_response(result, completion_time)
            yield {"status": "processing", "message": "Result uploaded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error uploading result: {str(e)}"}
            return

        # Final response
        yield {
            "status": "completed",
            "output": response
        }

    except Exception as e:
        yield {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
    })