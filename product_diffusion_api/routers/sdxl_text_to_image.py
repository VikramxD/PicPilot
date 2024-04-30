from diffusers import DiffusionPipeline
import torch
from fastapi import APIRouter
from pydantic import BaseModel
import json
import base64
from PIL import Image
from io import BytesIO



router = APIRouter()



def pil_to_b64_json(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    json_data = {"b64_image": b64_image}
    return json_data


class SDXLLoraInference:
    """
    Class for running inference using the SDXL-LoRA model to generate stunning product photographs.

    Args:
        prompt (str): The input prompt for generating the product photograph.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale factor for guidance during inference.
    """

    def __init__(
        self, prompt: str, negative_prompt:str,num_images:int ,num_inference_steps: int, guidance_scale: float
    ) -> None:
        self.model_path = "VikramSingh178/sdxl-lora-finetune-product-caption"
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(self.model_path)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images

    def run_inference(self):
        """
        Runs inference using the SDXL-LoRA model to generate a stunning product photograph.

        Returns:
            images: The generated product photograph(s).
        """

        prompt = self.prompt
        negative_prompt = self.negative_prompt
        num_images = self.num_images
        
        image =  self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=num_images
        ).images[0]
        image_json = pil_to_b64_json(image)
        return image_json

   

class InputFormat(BaseModel):
    prompt : str
    negative_prompt : str
    num_images : int
    num_inference_steps : int
    guidance_scale : float
    



@router.post("/sdxl_v0_lora_inference")
async def sdxl_v0_lora_inference(data: InputFormat):
    """
    Perform SDXL V0 LoRa inference.

    Args:
        data (InputFormat): The input data containing the prompt, number of inference steps, and guidance scale.

    Returns:
        The output of the inference.
    """
    prompt = data.prompt
    negative_prompt = data.negative_prompt,
    num_images = data.num_images
    num_inference_steps = data.num_inference_steps
    guidance_scale = data.guidance_scale
    inference = SDXLLoraInference(prompt,negative_prompt, num_inference_steps, guidance_scale,num_images)
    output_json = inference.run_inference()
    return output_json
   

    
    
    