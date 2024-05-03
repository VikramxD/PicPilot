from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from typing import List
import uuid
from diffusers import DiffusionPipeline
import torch

router = APIRouter()

# Utility function to convert PIL image to base64 encoded JSON
def pil_to_b64_json(image):
    # Generate a UUID for the image
    image_id = str(uuid.uuid4())
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_id": image_id, "b64_image": b64_image}


# SDXLLoraInference class for running inference
class SDXLLoraInference:
    """
    Class for performing SDXL Lora inference.

    Args:
        prompt (str): The prompt for generating the image.
        negative_prompt (str): The negative prompt for generating the image.
        num_images (int): The number of images to generate.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale for guiding the generation process.

    Attributes:
        pipe (DiffusionPipeline): The pre-trained diffusion pipeline.
        prompt (str): The prompt for generating the image.
        negative_prompt (str): The negative prompt for generating the image.
        num_images (int): The number of images to generate.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale for guiding the generation process.

    Methods:
        run_inference: Runs the inference process and returns the generated image.
    """

    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        num_images: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        self.model_path = "VikramSingh178/sdxl-lora-finetune-product-caption"
        self.pipe.load_lora_weights(self.model_path)
        self.pipe.to('cuda')
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def run_inference(self):
        """
        Runs the inference process and returns the generated image.

        Returns:
            str: The generated image in base64-encoded JSON format.
        """
        image = self.pipe(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=self.num_images,
        ).images[0]
        return pil_to_b64_json(image)

# Input format for single request
class InputFormat(BaseModel):
    prompt: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str
    num_images: int

# Input format for batch requests
class BatchInputFormat(BaseModel):
    batch_input: List[InputFormat]

# Endpoint for single request
@router.post("/sdxl_v0_lora_inference")
async def sdxl_v0_lora_inference(data: InputFormat):
    inference = SDXLLoraInference(
        data.prompt,
        data.negative_prompt,
        data.num_images,
        data.num_inference_steps,
        data.guidance_scale,
    )
    output_json = inference.run_inference()
    return output_json

# Endpoint for batch requests
@router.post("/sdxl_v0_lora_inference/batch")
async def sdxl_v0_lora_inference_batch(data: BatchInputFormat):
    """
    Perform batch inference for SDXL V0 LoRa model.

    Args:
        data (BatchInputFormat): The input data containing a batch of requests.

    Returns:
        dict: A dictionary containing the message and processed requests data.

    Raises:
        HTTPException: If the number of requests exceeds the maximum queue size.
    """
    MAX_QUEUE_SIZE = 32

    if len(data.batch_input) > MAX_QUEUE_SIZE:
        raise HTTPException(status_code=400, detail=f"Number of requests exceeds maximum queue size ({MAX_QUEUE_SIZE})")

    processed_requests = []
    for item in data.batch_input:
        inference = SDXLLoraInference(
            item.prompt,
            item.negative_prompt,
            item.num_images,
            item.num_inference_steps,
            item.guidance_scale,
        )
        output_json = inference.run_inference()
        processed_requests.append(output_json)

    return {"message": "Requests processed successfully", "data": processed_requests}
