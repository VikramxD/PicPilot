import torch
from diffusers import DiffusionPipeline
import litserve as ls
from typing import Dict, Any, List
from PIL import Image
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.api_utils import pil_to_b64_json, pil_to_s3_json

DEVICE = 'cuda:1'

class SDXLLoraAPI(ls.LitAPI):
    """
    LitAPI implementation for serving SDXL (Stable Diffusion XL) model with LoRA.

    This class defines the API for the SDXL model with LoRA, including methods for
    setup, request decoding, batching, prediction, and response encoding.
    """

    def setup(self, device: str) -> None:
        """
        Set up the SDXL pipeline with LoRA and optimize it for inference.

        Args:
            device (str): The device to run the model on (e.g., 'cuda:1').
        """
        self.device = device
        self.sdxl_pipeline = DiffusionPipeline.from_pretrained(
            tti_settings.MODEL_NAME, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.sdxl_pipeline.load_lora_weights(tti_settings.ADAPTER_NAME)
        self.sdxl_pipeline.fuse_lora()
        self.sdxl_pipeline.unet.to(memory_format=torch.channels_last)
        if tti_settings.ENABLE_COMPILE:
            self.sdxl_pipeline.unet = torch.compile(self.sdxl_pipeline.unet, mode="max-autotune")
            self.sdxl_pipeline.vae.decode = torch.compile(self.sdxl_pipeline.vae.decode, mode="max-autotune")
        self.sdxl_pipeline.fuse_qkv_projections()

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request and prepare inputs for the model.

        Args:
            request (Dict[str, Any]): The raw request data.

        Returns:
            Dict[str, Any]: The decoded request with processed inputs.
        """
        return {
            "prompt": request["prompt"],
            "negative_prompt": request.get("negative_prompt", ""),
            "num_images": request.get("num_images", 1),
            "num_inference_steps": request.get("num_inference_steps", 50),
            "guidance_scale": request.get("guidance_scale", 7.5),
            "mode": request.get("mode", "s3_json")
        }

    def batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Batch multiple inputs together for efficient processing.

        Args:
            inputs (List[Dict[str, Any]]): A list of individual inputs.

        Returns:
            Dict[str, List[Any]]: A dictionary of batched inputs.
        """
        return {
            "prompt": [input["prompt"] for input in inputs],
            "negative_prompt": [input["negative_prompt"] for input in inputs],
            "num_images": [input["num_images"] for input in inputs],
            "num_inference_steps": [input["num_inference_steps"] for input in inputs],
            "guidance_scale": [input["guidance_scale"] for input in inputs],
            "mode": [input["mode"] for input in inputs]
        }

    def predict(self, inputs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Run predictions on the batched inputs.

        Args:
            inputs (Dict[str, List[Any]]): Batched inputs for the model.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing generated images and their modes.
        """
        total_images = sum(inputs["num_images"])
        images = self.sdxl_pipeline(
            prompt=inputs["prompt"],
            negative_prompt=inputs["negative_prompt"],
            num_images_per_prompt=1,  # Generate one image per prompt
            num_inference_steps=inputs["num_inference_steps"][0],  # Use the first value
            guidance_scale=inputs["guidance_scale"][0],  # Use the first value
        ).images

        # Repeat images based on num_images and pair with modes
        results = []
        for img, num, mode in zip(images, inputs["num_images"], inputs["mode"]):
            results.extend([{"image": img, "mode": mode} for _ in range(num)])

        return results[:total_images]

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Unbatch the outputs from the predict method.

        Args:
            outputs (List[Dict[str, Any]]): The batched outputs from predict.

        Returns:
            List[Dict[str, Any]]: The unbatched list of outputs.
        """
        return outputs

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode the model output into a response payload.

        Args:
            output (Dict[str, Any]): The generated image and its mode.

        Returns:
            Dict[str, Any]: The encoded response with either S3 URL or base64 encoded image.
        """
        mode = output["mode"]
        image = output["image"]
        if mode == "s3_json":
            return pil_to_s3_json(image, "sdxl_image")
        elif mode == "b64_json":
            return pil_to_b64_json(image)
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

if __name__ == "__main__":
    api = SDXLLoraAPI()
    #server = ls.LitServer(
         #   api,
         #   accelerator="auto",
         #   max_batch_size=tti_settings.MAX_BATCH_SIZE,
         #   batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6,
       # )
    #server.run(port=8000)