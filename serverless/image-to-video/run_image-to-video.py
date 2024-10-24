import io
import os
import runpod
import tempfile
import time
from typing import Dict, Any, List, Union, Tuple
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

device = "cuda"
pipeline = ImageToVideoPipeline(device=device)

def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming video generation request.
    
    Args:
        request: Raw request data containing base64 image and parameters
        
    Returns:
        Dict containing decoded PIL Image and validated parameters
        
    Raises:
        ValueError: If request validation or image decoding fails
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

def generate_frames(inputs: Dict[str, Any]) -> Tuple[Union[List[Image.Image], Image.Image], float]:
    """
    Generate video frames using the pipeline.
    
    Args:
        inputs: Dictionary containing input image and generation parameters
        
    Returns:
        Tuple containing generated frames and completion time
    """
    start_time = time.time()
    
    frames = pipeline.generate(
        prompt=inputs['params']['prompt'],
        image=inputs['image'],
        num_frames=inputs['params']['num_frames'],
        num_inference_steps=inputs['params']['num_inference_steps'],
        guidance_scale=inputs['params']['guidance_scale'],
        height=inputs['params']['height'],
        width=inputs['params']['width'],
        use_dynamic_cfg=inputs['params']['use_dynamic_cfg']
    )
    
    if isinstance(frames, tuple):
        frames = frames[0]
    elif hasattr(frames, 'frames'):
        frames = frames.frames[0]
        
    completion_time = time.time() - start_time
    return frames, completion_time

def create_video_response(frames: List[Image.Image], completion_time: float, fps: int) -> Dict[str, Any]:
    """
    Create video file and generate response with S3 URL.
    
    Args:
        frames: List of generated video frames
        completion_time: Time taken for generation
        fps: Frames per second for the output video
        
    Returns:
        Dict containing S3 URL, completion time, and video metadata
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, "generated_video.mp4")
        export_to_video(frames, temp_video_path, fps=fps)
        
        with open(temp_video_path, "rb") as video_file:
            s3_response = mp4_to_s3_json(
                video_file, 
                f"generated_video_{int(time.time())}.mp4"
            )
            
    return {
        "result": s3_response,
        "completion_time": round(completion_time, 2),
        "video_resolution": f"{frames[0].width}x{frames[0].height}",
        "fps": fps
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for processing video generation requests.
    
    Args:
        job: RunPod job dictionary containing input data
        
    Returns:
        Dict containing either the processed video result or error information
    """
    try:
        inputs = decode_request(job['input'])
        frames, completion_time = generate_frames(inputs)
        return create_video_response(frames, completion_time, inputs['params']['fps'])
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })