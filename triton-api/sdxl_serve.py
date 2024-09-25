# sdxl_lora_api.py

import torch
from diffusers import DiffusionPipeline
import litserve as ls
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.api_utils import accelerator, pil_to_b64_json, pil_to_s3_json

DEVICE = accelerator()

class SDXLLoraAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.sdxl_pipeline = DiffusionPipeline.from_pretrained(
            settings.MODEL_NAME, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.sdxl_pipeline.load_lora_weights(settings.ADAPTER_NAME)
        self.sdxl_pipeline.fuse_lora()
        self.sdxl_pipeline.unet.to(memory_format=torch.channels_last)
        if settings.ENABLE_COMPILE:
            self.sdxl_pipeline.unet = torch.compile(self.sdxl_pipeline.unet, mode="max-autotune")
            self.sdxl_pipeline.vae.decode = torch.compile(self.sdxl_pipeline.vae.decode, mode="max-autotune")
        self.sdxl_pipeline.fuse_qkv_projections()

    def decode_request(self, request):
        return {
            "prompt": request["prompt"],
            "negative_prompt": request.get("negative_prompt", ""),
            "num_images": request.get("num_images", 1),
            "num_inference_steps": request.get("num_inference_steps", 50),
            "guidance_scale": request.get("guidance_scale", 7.5),
            "mode": request.get("mode", "s3_json")
        }

    def predict(self, inputs):
        images = self.sdxl_pipeline(
            prompt=inputs["prompt"],
            negative_prompt=inputs["negative_prompt"],
            num_images_per_prompt=inputs["num_images"],
            num_inference_steps=inputs["num_inference_steps"],
            guidance_scale=inputs["guidance_scale"],
        ).images
        return images[0]  # Return the first image

    def encode_response(self, output):
        mode = self.context.get("mode", "s3_json")
        if mode == "s3_json":
            return pil_to_s3_json(output, "sdxl_image")
        elif mode == "b64_json":
            return pil_to_b64_json(output)
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

if __name__ == "__main__":
    api = SDXLLoraAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=tti_settings.MAX_BATCH_SIZE,
        batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6,
    )
    server.run(port=8001)