import io
import os
import tempfile
from typing import Dict, Any, Tuple
from PIL import Image
import base64
from pydantic import BaseModel, Field
import time
from diffusers.utils import export_to_video
from litserve import LitAPI, LitServer
from scripts.api_utils import mp4_to_s3_json
from scripts.image_to_video import ImageToVideoPipeline

class ImageToVideoRequest(BaseModel):
    """
    Pydantic model representing a request for image-to-video generation.
    """
    image: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., description="Text prompt for video generation")
    num_frames: int = Field(49, description="Number of frames to generate")
    num_inference_steps: int = Field(50, description="Number of inference steps")
    guidance_scale: float = Field(6.0, description="Guidance scale")
    height: int = Field(480, description="Height of the output video")
    width: int = Field(720, description="Width of the output video")
    use_dynamic_cfg: bool = Field(True, description="Use dynamic CFG")
    fps: int = Field(30, description="Frames per second for the output video")

class ImageToVideoAPI(LitAPI):
    """
    LitAPI implementation for Image-to-Video model serving.
    """

    def setup(self, device: str) -> None:
        """
        Set up the Image-to-Video pipeline and associated resources.
        """
        self.device = device
        self.pipeline = ImageToVideoPipeline(device=device)

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request and prepare inputs for the model.
        """
        try:
            video_request = ImageToVideoRequest(**request)
            image_data = base64.b64decode(video_request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            return {
                'image': image,
                'params': video_request.model_dump()
            }
        except Exception as e:
            raise ValueError(f"Invalid request: {str(e)}")

    def predict(self, inputs: Dict[str, Any]) -> Tuple[list, float, int]:
        """
        Run predictions on the input.
        """
        image = inputs['image']
        params = inputs['params']

        start_time = time.time()

        result = self.pipeline.generate(
            prompt=params['prompt'],
            image=image,
            num_frames=params['num_frames'],
            num_inference_steps=params['num_inference_steps'],
            guidance_scale=params['guidance_scale'],
            height=params['height'],
            width=params['width'],
            use_dynamic_cfg=params['use_dynamic_cfg']
        )

        if isinstance(result, tuple):
            frames = result[0]
        elif hasattr(result, 'frames'):
            frames = result.frames
        else:
            frames = result

        completion_time = time.time() - start_time
        return frames, completion_time, params['fps']

    def encode_response(self, output: Tuple[list, float, int]) -> Dict[str, Any]:
        """
        Encode the model output and additional information into a response payload.
        """
        frames, completion_time, fps = output
        try:
            # Create a temporary directory to store the video file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video_path = os.path.join(temp_dir, "generated_video.mp4")
                
                # Export the video to the temporary file
                export_to_video(frames, temp_video_path, fps=fps)
                
                # Read the video file and upload to S3
                with open(temp_video_path, "rb") as video_file:
                    s3_response = mp4_to_s3_json(video_file, "generated_video.mp4")
            
            return {
                "result": s3_response,
                "completion_time": round(completion_time, 2),
                "video_resolution": f"{frames[0].width}x{frames[0].height}",
                "fps": fps
            }
        except Exception as e:
            # Log the error for debugging
            print(f"Error in encode_response: {str(e)}")
            raise

if __name__ == "__main__":
    api = ImageToVideoAPI()
    server = LitServer(api, accelerator="cuda", max_batch_size=1)
    server.run(port=8000)