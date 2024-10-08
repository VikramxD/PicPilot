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
from scripts.flux_inference import FluxInpaintingInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_manager = S3ManagerService()

class InpaintingRequest(BaseModel):
    """
    Model representing an inpainting request.

    Attributes:
        prompt (str): The prompt for inpainting.
        strength (float): Strength of inpainting effect, between 0.0 and 1.0.
        seed (int): Random seed for reproducibility.
        num_inference_steps (int): Number of inference steps, between 1 and 1000.
        input_image (str): Base64 encoded input image.
        mask_image (str): Base64 encoded mask image.
    """
    prompt: str = Field(..., description="The prompt for inpainting")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Strength of inpainting effect")
    seed: int = Field(42, description="Random seed for reproducibility")
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Number of inference steps")
    input_image: str = Field(..., description="Base64 encoded input image")
    mask_image: str = Field(..., description="Base64 encoded mask image")

class FluxInpaintingAPI(LitAPI):
    """
    API for Flux Inpainting using LitServer.

    This class implements the LitAPI interface to provide inpainting functionality
    using the Flux Inpainting model. It handles request decoding, batching,
    prediction, and response encoding.
    """

    def setup(self, device: str) -> None:
        """
        Initialize the Flux Inpainting model.

        Args:
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.flux_inpainter = FluxInpaintingInference()
        self.device = device

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request into a format suitable for processing.

        Args:
            request (Dict[str, Any]): The raw incoming request data.

        Returns:
            Dict[str, Any]: A dictionary containing the decoded request data.

        Raises:
            Exception: If there's an error in decoding the request.
        """
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
        """
        Prepare a batch of inputs for processing.

        Args:
            inputs (List[Dict[str, Any]]): A list of individual input dictionaries.

        Returns:
            Dict[str, List[Any]]: A dictionary containing batched inputs.
        """
        return {
            "prompt": [input["prompt"] for input in inputs],
            "input_image": [input["input_image"] for input in inputs],
            "mask_image": [input["mask_image"] for input in inputs],
            "strength": [input["strength"] for input in inputs],
            "seed": [input["seed"] for input in inputs],
            "num_inference_steps": [input["num_inference_steps"] for input in inputs]
        }

    def predict(self, inputs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of inputs and return the results.

        Args:
            inputs (Dict[str, List[Any]]): A dictionary containing batched inputs.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the prediction results.
        """
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
        """
        Convert batched outputs back to individual results.

        Args:
            outputs (List[Dict[str, Any]]): A list of output dictionaries from the predict method.

        Returns:
            List[Dict[str, Any]]: The same list of output dictionaries.
        """
        return outputs

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode the output image and prepare the response.

        Args:
            output (Dict[str, Any]): A dictionary containing the prediction output.

        Returns:
            Dict[str, Any]: A dictionary containing the encoded response with the result URL,
                            prompt, seed, and time taken.

        Raises:
            Exception: If there's an error in encoding the response.
        """
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
        api_path='/api/v2/inpainting/flux',
        accelerator="auto",
        max_batch_size=4,
        batch_timeout=0.1
    )
    server.run(port=8000)