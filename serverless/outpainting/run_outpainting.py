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

    This model defines the structure and validation rules for incoming API requests.
    All fields are required unless otherwise specified.
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

class OutpaintingService:
    """
    Service class for handling outpainting operations.
    Based on LitAPI implementation but adapted for RunPod.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize the outpainting service."""
        self.device = device
        self.outpainter = Outpainter()

    async def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request and prepare inputs for the model.

        Args:
            request: The raw request data.

        Returns:
            Dict containing decoded image and request parameters.

        Raises:
            ValueError: If request is invalid or cannot be processed.
        """
        try:
            outpainting_request = OutpaintingRequest(**request)
            # Run decode in thread pool
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

    async def predict(self, inputs: Dict[str, Any]) -> Tuple[Image.Image, float]:
        """
        Run predictions on the input.

        Args:
            inputs: Dict containing image and outpainting parameters.

        Returns:
            Tuple containing the resulting image and completion time.
        """
        image = inputs['image']
        params = inputs['params']

        start_time = time.time()

        # Run outpainting in thread pool
        result = await asyncio.to_thread(
            self.outpainter.outpaint,
            image,
            params['width'],
            params['height'],
            params['overlap_percentage'],
            params['num_inference_steps'],
            params['resize_option'],
            params['custom_resize_percentage'],
            params['prompt_input'],
            params['alignment'],
            params['overlap_left'],
            params['overlap_right'],
            params['overlap_top'],
            params['overlap_bottom']
        )

        completion_time = time.time() - start_time
        return result, completion_time

    async def encode_response(self, output: Tuple[Image.Image, float]) -> Dict[str, Any]:
        """
        Encode the model output into a response payload.

        Args:
            output: Tuple containing outpainted image and completion time.

        Returns:
            Dict containing S3 URL and metadata.
        """
        image, completion_time = output
        # Run S3 upload in thread pool
        img_str = await asyncio.to_thread(pil_to_s3_json, image, "outpainting_image")
        
        return {
            "result": img_str,
            "completion_time": round(completion_time, 2),
            "image_resolution": f"{image.width}x{image.height}"
        }

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod with progress updates.
    """
    try:
        # Create service instance
        service = OutpaintingService(device="cuda")
        yield {"status": "starting", "message": "Service initialized"}

        # Decode request
        try:
            inputs = await service.decode_request(job['input'])
            yield {
                "status": "processing", 
                "message": "Request decoded successfully",
                "input_resolution": f"{inputs['image'].width}x{inputs['image'].height}"
            }
        except Exception as e:
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Generate prediction
        try:
            yield {"status": "processing", "message": "Starting outpainting"}
            result = await service.predict(inputs)
            yield {
                "status": "processing", 
                "message": "Outpainting completed",
                "completion_time": f"{result[1]:.2f}s"
            }
        except Exception as e:
            yield {"status": "error", "message": f"Error during outpainting: {str(e)}"}
            return

        # Encode response
        try:
            yield {"status": "processing", "message": "Encoding result"}
            response = await service.encode_response(result)
            yield {"status": "processing", "message": "Result encoded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error encoding result: {str(e)}"}
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
    import runpod
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
        
    })