from pydantic import BaseModel
from fastapi import Form


class InpaintingRequest(BaseModel):
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    strength: float
    guidance_scale: float
    
