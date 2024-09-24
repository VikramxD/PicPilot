import io
import json
import base64
import time
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from PIL import Image
from litserve import LitAPI, LitServer
from scripts.s3_manager import S3ManagerService
from config_settings import settings
from flux_inference import FluxInpaintingInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_manager = S3ManagerService()

class InpaintingRequest(BaseModel):
    """Model representing an inpainting request."""
    prompt: str = Field(..., description="The prompt for inpainting")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Strength of inpainting effect")
    seed: int = Field(42, description="Random seed for reproducibility")
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Number of inference steps")
    input_image: str = Field(..., description="Base64 encoded input image")
    mask_image: str = Field(..., description="Base64 encoded mask image")

class FluxInpaintingAPI(LitAPI):
    """API for Flux Inpainting using LitServer."""

    def setup(self, device: str) -> None:
        """Initialize the Flux Inpainting model."""
        self.flux_inpainter = FluxInpaintingInference()
        self.device = device

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the incoming request into a format suitable for processing."""
        try:
            inpainting_request = InpaintingRequest(**request)

            input_image = Image.open(io.BytesIO(base64.b64decode(inpainting_request.input_image)))
            mask_image = Image.open(io.BytesIO(base64.b64decode(inpainting_request.mask_image)))

            return {
                "prompt": inpainting_request.prompt,
                "input_image": input_image,
                "mask_image": mask_image,
                "strength": inpainting_request.strength,
                "seed": inpainting_request.seed,
                "num_inference_steps": inpainting_request.num_inference_steps
            }
        except Exception as e:
            logger.error(f"Error in decode_request: {e}")
            raise

    def batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Prepare a batch of inputs for processing."""
        return {
            "prompt": [input["prompt"] for input in inputs],
            "input_image": [input["input_image"] for input in inputs],
            "mask_image": [input["mask_image"] for input in inputs],
            "strength": [input["strength"] for input in inputs],
            "seed": [input["seed"] for input in inputs],
            "num_inference_steps": [input["num_inference_steps"] for input in inputs]
        }

    def predict(self, inputs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs and return the results."""
        results = []
        for i in range(len(inputs["prompt"])):
            start_time = time.time()
            try:
                result_image = self.flux_inpainter.generate_inpainting(
                    input_image=inputs["input_image"][i],
                    mask_image=inputs["mask_image"][i],
                    prompt=inputs["prompt"][i],
                    seed=inputs["seed"][i],
                    strength=inputs["strength"][i],
                    num_inference_steps=inputs["num_inference_steps"][i]
                )
                end_time = time.time()
                results.append({
                    "image": result_image,
                    "prompt": inputs["prompt"][i],
                    "seed": inputs["seed"][i],
                    "time_taken": end_time - start_time
                })
            except Exception as e:
                logger.error(f"Error in predict for item {i}: {e}")
                results.append(None)
        return results

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert batched outputs back to individual results."""
        return outputs

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the output image and prepare the response."""
        if output is None:
            return {"error": "Failed to generate image"}

        try:
            result_image = output["image"]
            buffered = io.BytesIO()
            result_image.save(buffered, format="PNG")
            
            unique_filename = s3_manager.generate_unique_file_name("result.png")
            s3_manager.upload_file(io.BytesIO(buffered.getvalue()), unique_filename)
            signed_url = s3_manager.generate_signed_url(unique_filename, exp=43200)
            
            return {
                "result_url": signed_url,
                "prompt": output["prompt"],
                "seed": output["seed"],
                "time_taken": output["time_taken"]
            }
        except Exception as e:
            logger.error(f"Error in encode_response: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    api = FluxInpaintingAPI()
    server = LitServer(
        api,
        accelerator="auto",
        max_batch_size=4,
        batch_timeout=0.1,
        timeout = 10000
    )
    server.run(port=8000)