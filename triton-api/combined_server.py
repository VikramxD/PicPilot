import numpy as np
from typing import List, Dict
import io
from PIL import Image, UnidentifiedImageError
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from scripts.outpainting import ControlNetZoeDepthOutpainting
from scripts.api_utils import pil_to_b64_json
import logging

logger = logging.getLogger(__name__)

class OutpaintingModel:
    def __init__(self):
        self.pipeline = ControlNetZoeDepthOutpainting(target_size=(1024, 1024))

    def _infer_fn(self, image_bytes: np.ndarray) -> List[Image.Image]:
        images = []
        for img_bytes in image_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
            except UnidentifiedImageError as e:
                logger.error(f"Unidentified image error: {str(e)}")
                raise e
        return images

    @batch
    def outpaint(self, 
                 image: np.ndarray, 
                 base_image_description: np.ndarray,
                 base_image_negative_prompt: np.ndarray,
                 controlnet_conditioning_scale: np.ndarray,
                 controlnet_guidance_scale: np.ndarray,
                 controlnet_num_inference_steps: np.ndarray,
                 controlnet_guidance_end: np.ndarray,
                 background_extension_prompt: np.ndarray,
                 outpainting_negative_prompt: np.ndarray,
                 outpainting_guidance_scale: np.ndarray,
                 outpainting_strength: np.ndarray,
                 outpainting_num_inference_steps: np.ndarray) -> Dict[str, np.ndarray]:
        results = []
        images = self._infer_fn(image)

        for i, img in enumerate(images):
            try:
                result = self.pipeline.run_pipeline(
                    img,
                    controlnet_prompt=base_image_description[i].decode('utf-8'),
                    controlnet_negative_prompt=base_image_negative_prompt[i].decode('utf-8'),
                    controlnet_conditioning_scale=float(controlnet_conditioning_scale[i]),
                    controlnet_guidance_scale=float(controlnet_guidance_scale[i]),
                    controlnet_num_inference_steps=int(controlnet_num_inference_steps[i]),
                    controlnet_guidance_end=float(controlnet_guidance_end[i]),
                    inpainting_prompt=background_extension_prompt[i].decode('utf-8'),
                    inpainting_negative_prompt=outpainting_negative_prompt[i].decode('utf-8'),
                    inpainting_guidance_scale=float(outpainting_guidance_scale[i]),
                    inpainting_strength=float(outpainting_strength[i]),
                    inpainting_num_inference_steps=int(outpainting_num_inference_steps[i])
                )

                result_json = pil_to_b64_json(result)
                results.append(result_json.encode('utf-8'))
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append(str(e).encode('utf-8'))
        
        return {"result": np.array(results)}

# Initialize and serve the model
model = OutpaintingModel()

triton = Triton()
triton.bind(
    model_name="outpainting",
    infer_func=model.outpaint,
    inputs=[
        Tensor(name="image", dtype=np.bytes_, shape=(-1,)),
        Tensor(name="base_image_description", dtype=np.bytes_, shape=(-1, 1)),
        Tensor(name="base_image_negative_prompt", dtype=np.bytes_, shape=(-1, 1)),
        Tensor(name="controlnet_conditioning_scale", dtype=np.float32, shape=(-1, 1)),
        Tensor(name="controlnet_guidance_scale", dtype=np.float32, shape=(-1, 1)),
        Tensor(name="controlnet_num_inference_steps", dtype=np.int32, shape=(-1, 1)),
        Tensor(name="controlnet_guidance_end", dtype=np.float32, shape=(-1, 1)),
        Tensor(name="background_extension_prompt", dtype=np.bytes_, shape=(-1, 1)),
        Tensor(name="outpainting_negative_prompt", dtype=np.bytes_, shape=(-1, 1)),
        Tensor(name="outpainting_guidance_scale", dtype=np.float32, shape=(-1, 1)),
        Tensor(name="outpainting_strength", dtype=np.float32, shape=(-1, 1)),
        Tensor(name="outpainting_num_inference_steps", dtype=np.int32, shape=(-1, 1))
    ],
    outputs=[
        Tensor(name="result", dtype=np.bytes_, shape=(-1,))
    ],
    config=ModelConfig(max_batch_size=8)
)

triton.serve()
