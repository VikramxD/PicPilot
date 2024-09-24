import io
import base64
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import litserve as ls
from scripts.s3_manager import S3ManagerService
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.flux_inference import FluxInpaintingInference
from scripts.api_utils import accelerator, pil_to_b64_json, pil_to_s3_json

s3_manager = S3ManagerService()
DEVICE = accelerator()

class MultiModelAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.flux_inpainter = FluxInpaintingInference()
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

    def decode_request(self, request):
        model_type = request.get("model_type")
        if model_type == "flux_inpainting":
            input_image = Image.open(io.BytesIO(base64.b64decode(request["input_image"])))
            mask_image = Image.open(io.BytesIO(base64.b64decode(request["mask_image"])))
            return {
                "model_type": model_type,
                "prompt": request["prompt"],
                "input_image": input_image,
                "mask_image": mask_image,
                "strength": request["strength"],
                "seed": request["seed"],
                "num_inference_steps": request["num_inference_steps"]
            }
        elif model_type == "sdxl_lora":
            return {
                "model_type": model_type,
                "prompt": request["prompt"],
                "negative_prompt": request.get("negative_prompt", ""),
                "num_images": request.get("num_images", 1),
                "num_inference_steps": request.get("num_inference_steps", 50),
                "guidance_scale": request.get("guidance_scale", 7.5),
                "mode": request.get("mode", "s3_json")
            }
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    def predict(self, inputs):
        model_type = inputs["model_type"]
        if model_type == "flux_inpainting":
            return self.flux_inpainter.generate_inpainting(
                input_image=inputs["input_image"],
                mask_image=inputs["mask_image"],
                prompt=inputs["prompt"],
                seed=inputs["seed"],
                strength=inputs["strength"],
                num_inference_steps=inputs["num_inference_steps"]
            )
        elif model_type == "sdxl_lora":
            images = self.sdxl_pipeline(
                prompt=inputs["prompt"],
                negative_prompt=inputs["negative_prompt"],
                num_images_per_prompt=inputs["num_images"],
                num_inference_steps=inputs["num_inference_steps"],
                guidance_scale=inputs["guidance_scale"],
            ).images
            return images  # Return the first image

    def encode_response(self, output):
        model_type = self.context.get("model_type")
        if model_type == "flux_inpainting":
            buffered = io.BytesIO()
            output.save(buffered, format="PNG")
            unique_filename = s3_manager.generate_unique_file_name("result.png")
            s3_manager.upload_file(io.BytesIO(buffered.getvalue()), unique_filename)
            signed_url = s3_manager.generate_signed_url(unique_filename, exp=43200)
            return {
                "result_url": signed_url,
                "prompt": self.context.get("prompt"),
                "seed": self.context.get("seed")
            }
        elif model_type == "sdxl_lora":
            mode = self.context.get("mode", "s3_json")
            if mode == "s3_json":
                return pil_to_s3_json(output, "sdxl_image")
            elif mode == "b64_json":
                return pil_to_b64_json(output)
            else:
                raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

if __name__ == "__main__":
    api = MultiModelAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=tti_settings.MAX_BATCH_SIZE,
        batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6,
        timeout = 10000
    )
    server.run(port=8000)