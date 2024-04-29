<<<<<<< HEAD
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image






def fetch_video_pipeline(video_model_name):
    """
    Fetches the video pipeline for image processing.

    Args:
        video_model_name (str): The name of the video model.

    Returns:
        pipe (StableVideoDiffusionPipeline): The video pipeline.

    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        video_model_name, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to('cuda')
    return pipe
=======
>>>>>>> aaed2f5 (Refactor config.py and models.py, and add new functions in segment_everything.py and video_pipeline.py)
