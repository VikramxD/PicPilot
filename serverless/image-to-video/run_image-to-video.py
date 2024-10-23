import io
import os
import runpod
import tempfile
import time
from typing import Dict, Any
from PIL import Image
import base64
from pydantic import BaseModel, Field
from diffusers.utils import export_to_video
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

class RunPodHandler:
    def __init__(self):
        self.device = "cuda"
        self.pipeline = ImageToVideoPipeline(device=self.device)

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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

    def generate_video(self, job: Dict[str, Any]) -> Dict[str, Any]:
        try:
            inputs = self.decode_request(job['input'])
            image = inputs['image']
            params = inputs['params']

            start_time = time.time()

            # Generate frames using the pipeline
            frames = self.pipeline.generate(
                prompt=params['prompt'],
                image=image,
                num_frames=params['num_frames'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                height=params['height'],
                width=params['width'],
                use_dynamic_cfg=params['use_dynamic_cfg']
            )
            
            if isinstance(frames, tuple):
                frames = frames[0]
            elif hasattr(frames, 'frames'):
                frames = frames.frames[0]

            completion_time = time.time() - start_time
            fps = params['fps']

            # Create temporary video file and upload to S3
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video_path = os.path.join(temp_dir, "generated_video.mp4")
                export_to_video(frames, temp_video_path, fps=fps)
                
                with open(temp_video_path, "rb") as video_file:
                    s3_response = mp4_to_s3_json(video_file, f"generated_video_{int(time.time())}.mp4")

            return {
                "result": s3_response,
                "completion_time": round(completion_time, 2),
                "video_resolution": f"{frames[0].width}x{frames[0].height}",
                "fps": fps
            }

        except Exception as e:
            return {"error": str(e)}

def handler(job):
    """
    RunPod handler function.
    """
    handler = RunPodHandler()
    return handler.generate_video(job)

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })