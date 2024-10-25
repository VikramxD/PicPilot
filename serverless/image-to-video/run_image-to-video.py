import io
import os
import runpod
import tempfile
import time
import asyncio
from typing import Dict, Any, List, Union, Tuple, AsyncGenerator
from PIL import Image
import base64
from pydantic import BaseModel, Field
from diffusers.utils import export_to_video
from scripts.api_utils import mp4_to_s3_json
from scripts.image_to_video import ImageToVideoPipeline

# Global pipeline instance
global_pipeline = None

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

async def initialize_pipeline():
    """Initialize the pipeline if not already loaded"""
    global global_pipeline
    if global_pipeline is None:
        print("Initializing Image to Video pipeline...")
        global_pipeline = ImageToVideoPipeline(device="cuda")
        print("Pipeline initialized successfully")
    return global_pipeline

async def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming video generation request asynchronously.
    """
    try:
        video_request = ImageToVideoRequest(**request)
        # Run decode in thread pool
        image_data = await asyncio.to_thread(base64.b64decode, video_request.image)
        image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(image_data)).convert("RGB")
        )
        
        return {
            'image': image,
            'params': video_request.model_dump()
        }
    except Exception as e:
        raise ValueError(f"Invalid request: {str(e)}")

async def generate_frames(inputs: Dict[str, Any], pipeline: ImageToVideoPipeline) -> Tuple[List[Image.Image], float]:
    """
    Generate video frames using the pipeline asynchronously.
    """
    start_time = time.time()
    
    # Run generation in thread pool
    frames = await asyncio.to_thread(
        pipeline.generate,
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

async def create_video_response(frames: List[Image.Image], completion_time: float, fps: int) -> Dict[str, Any]:
    """
    Create video file and generate response with S3 URL asynchronously.
    """
    def create_video():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, "generated_video.mp4")
            export_to_video(frames, temp_video_path, fps=fps)
            
            with open(temp_video_path, "rb") as video_file:
                return mp4_to_s3_json(
                    video_file, 
                    f"generated_video_{int(time.time())}.mp4"
                )
    
    # Run video creation and upload in thread pool
    s3_response = await asyncio.to_thread(create_video)
            
    return {
        "result": s3_response,
        "completion_time": round(completion_time, 2),
        "video_resolution": f"{frames[0].width}x{frames[0].height}",
        "fps": fps
    }

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod with progress updates.
    """
    try:
        # Initial status
        yield {"status": "starting", "message": "Initializing video generation process"}

        # Initialize pipeline
        pipeline = await initialize_pipeline()
        yield {"status": "processing", "message": "Pipeline loaded successfully"}

        # Decode request
        try:
            inputs = await decode_request(job['input'])
            yield {"status": "processing", "message": "Request decoded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Generate frames with progress updates
        try:
            yield {"status": "processing", "message": "Generating video frames"}
            frames, completion_time = await generate_frames(inputs, pipeline)
            yield {"status": "processing", "message": f"Generated {len(frames)} frames successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error generating frames: {str(e)}"}
            return

        # Create and upload video
        try:
            yield {"status": "processing", "message": "Creating and uploading video"}
            response = await create_video_response(
                frames, 
                completion_time, 
                inputs['params']['fps']
            )
            yield {"status": "processing", "message": "Video uploaded successfully"}
        except Exception as e:
            yield {"status": "error", "message": f"Error creating video: {str(e)}"}
            return

        # Final response
        yield {
            "status": "completed",
            "output": response
        }

    except Exception as e:
        yield {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def calculate_progress(current_frame: int, total_frames: int) -> dict:
    """Calculate progress percentage and create status update."""
    progress = (current_frame / total_frames) * 100
    return {
        "status": "processing",
        "progress": round(progress, 2),
        "message": f"Generating frame {current_frame}/{total_frames}"
    }

# Initialize the pipeline when the service starts
print("Initializing service...")
asyncio.get_event_loop().run_until_complete(initialize_pipeline())
print("Service initialization complete")

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
    })