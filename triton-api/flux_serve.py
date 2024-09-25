import io
import base64
from PIL import Image
import litserve as ls
from scripts.s3_manager import S3ManagerService
from config_settings import settings
from scripts.flux_inference import FluxInpaintingInference
from scripts.api_utils import accelerator
from configs.tti_settings import tti_settings
s3_manager = S3ManagerService()
DEVICE = accelerator()

class FluxInpaintingAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.flux_inpainter = FluxInpaintingInference()

    def decode_request(self, request):
        input_image = Image.open(io.BytesIO(base64.b64decode(request["input_image"])))
        mask_image = Image.open(io.BytesIO(base64.b64decode(request["mask_image"])))
        return {
            "prompt": request["prompt"],
            "input_image": input_image,
            "mask_image": mask_image,
            "strength": request["strength"],
            "seed": request["seed"],
            "num_inference_steps": request["num_inference_steps"]
        }

    def predict(self, inputs):
        return self.flux_inpainter.generate_inpainting(
            input_image=inputs["input_image"],
            mask_image=inputs["mask_image"],
            prompt=inputs["prompt"],
            seed=inputs["seed"],
            strength=inputs["strength"],
            num_inference_steps=inputs["num_inference_steps"]
        )

    def encode_response(self, output):
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

if __name__ == "__main__":
    api = FluxInpaintingAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=4,
        batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6,
    )
    server.run(port=8000)