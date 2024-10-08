from litserve import LitAPI, LitServer
from typing import Dict, Any, Tuple
from PIL import Image
import io
import base64
from pydantic import BaseModel, Field
import torch
import time
from scripts.outpainting import Outpainter
from scripts.api_utils import pil_to_s3_json

class OutpaintingRequest(BaseModel):
    """
    Pydantic model representing a request for outpainting inference.

    This model defines the structure and validation rules for incoming API requests.
    All fields are required unless otherwise specified.

    Attributes:
        image (str): Base64 encoded input image.
        width (int): Target width for the outpainted image.
        height (int): Target height for the outpainted image.
        overlap_percentage (int): Percentage of overlap for the mask.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        resize_option (str): Option for resizing the input image ("Full", "50%", "33%", "25%", or "Custom").
        custom_resize_percentage (int): Custom resize percentage when resize_option is "Custom".
        prompt_input (str): Text prompt to guide the outpainting process.
        alignment (str): Alignment of the original image within the new canvas.
        overlap_left (bool): Whether to apply overlap on the left side.
        overlap_right (bool): Whether to apply overlap on the right side.
        overlap_top (bool): Whether to apply overlap on the top side.
        overlap_bottom (bool): Whether to apply overlap on the bottom side.
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

class OutpaintingAPI(LitAPI):
    """
    LitAPI implementation for Outpainting model serving.

    This class defines the API for the Outpainting model, including methods for
    request decoding, prediction, and response encoding. It uses the Outpainter
    class to perform the actual outpainting operations.

    Attributes:
        outpainter (Outpainter): An instance of the Outpainter class for performing outpainting.

    Methods:
        setup: Initialize the Outpainter and set up any necessary resources.
        decode_request: Decode and validate incoming API requests.
        predict: Perform the outpainting operation on the input image.
        encode_response: Encode the outpainted image and additional information as a response.
    """

    def setup(self, device: str) -> None:
        """
        Set up the Outpainting model and associated resources.

        This method is called once when the API is initialized. It creates an instance
        of the Outpainter class and performs any necessary setup.

        Args:
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            None
        """
        self.device = device
        self.outpainter = Outpainter()

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request and prepare inputs for the model.

        This method validates the incoming request against the OutpaintingRequest model,
        decodes the base64 encoded image, and prepares the inputs for the outpainting process.

        Args:
            request (Dict[str, Any]): The raw request data.

        Returns:
            Dict[str, Any]: A dictionary containing the decoded image and request parameters.

        Raises:
            ValueError: If the request is invalid or cannot be processed.
        """
        try:
            outpainting_request = OutpaintingRequest(**request)
            image_data = base64.b64decode(outpainting_request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            
            return {
                'image': image,
                'params': outpainting_request.dict()
            }
        except Exception as e:
            raise ValueError(f"Invalid request: {str(e)}")

    def predict(self, inputs: Dict[str, Any]) -> Tuple[Image.Image, float, float]:
        """
        Run predictions on the input.

        This method performs the outpainting operation using the Outpainter instance.
        It takes the decoded inputs from decode_request and passes them to the outpainter.
        It also measures the completion time and calculates the prompt ratio.

        Args:
            inputs (Dict[str, Any]): A dictionary containing the image and outpainting parameters.

        Returns:
            Tuple[Image.Image, float, float]: A tuple containing:
                - The resulting outpainted image
                - The completion time in seconds
                - The prompt ratio (ratio of prompt tokens to total tokens)
        """
        image = inputs['image']
        params = inputs['params']

        start_time = time.time()

        result = self.outpainter.outpaint(
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

    def encode_response(self, output: Tuple[Image.Image, float, float]) -> Dict[str, Any]:
        """
        Encode the model output and additional information into a response payload.

        This method takes the outpainted image, completion time, and prompt ratio,
        encodes the image as a base64 string, and prepares the final API response
        with additional information.

        Args:
            output (Tuple[Image.Image, float, float]): A tuple containing:
                - The outpainted image produced by the predict method
                - The completion time in seconds
                - The prompt ratio

        Returns:
            Dict[str, Any]: A dictionary containing the base64 encoded image string,
                            completion time, prompt ratio, and image resolution.
        """
        image, completion_time = output
        img_str = pil_to_s3_json(image,"outpainting_image")
        
        return {
            "result": img_str,
            "completion_time": round(completion_time, 2),
            "image_resolution": f"{image.width}x{image.height}"
        }


   

if __name__ == "__main__":
    api = OutpaintingAPI()
    server = LitServer(api, accelerator="cuda", max_batch_size=1)
    server.run(port=8000)