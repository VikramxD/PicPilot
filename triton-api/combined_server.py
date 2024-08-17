import numpy as np
from typing import List, Dict, Any
import tempfile
import os
import io
import logging
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from scripts.outpainting import ControlNetZoeDepthOutpainting
from scripts.api_utils import pil_to_b64_json
from PIL import Image

logger = logging.getLogger(__name__)

class OutpaintingModel:
    def __init__(self):
        self.pipeline = ControlNetZoeDepthOutpainting(target_size=(1024, 1024))

    @staticmethod
    def _decode_string(value: np.ndarray) -> str:
        if isinstance(value[0], np.ndarray):
            return value[0][0].decode('utf-8')
        return value[0].decode('utf-8')

    @staticmethod
    def _get_float(value: np.ndarray) -> float:
        return float(value[0][0])

    @staticmethod
    def _get_int(value: np.ndarray) -> int:
        return int(value[0][0])

    def _infer_fn(self, image: np.ndarray) -> List[Image.Image]:
        logger.debug(f"Image data: {image.shape} ({image.size})")
        images = []
        for img in images:
            images.append(img)
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
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_filename = temp_file.name
                img.save(temp_filename)
            
            try:
                result = self.pipeline.run_pipeline(
                    temp_filename,
                    controlnet_prompt=self._decode_string(base_image_description[i]),
                    controlnet_negative_prompt=self._decode_string(base_image_negative_prompt[i]),
                    controlnet_conditioning_scale=self._get_float(controlnet_conditioning_scale[i]),
                    controlnet_guidance_scale=self._get_float(controlnet_guidance_scale[i]),
                    controlnet_num_inference_steps=self._get_int(controlnet_num_inference_steps[i]),
                    controlnet_guidance_end=self._get_float(controlnet_guidance_end[i]),
                    inpainting_prompt=self._decode_string(background_extension_prompt[i]),
                    inpainting_negative_prompt=self._decode_string(outpainting_negative_prompt[i]),
                    inpainting_guidance_scale=self._get_float(outpainting_guidance_scale[i]),
                    inpainting_strength=self._get_float(outpainting_strength[i]),
                    inpainting_num_inference_steps=self._get_int(outpainting_num_inference_steps[i])
                )
                
                result_json = pil_to_b64_json(result)
                results.append(result_json.encode('utf-8'))
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append(str(e).encode('utf-8'))
            finally:
                os.unlink(temp_filename)
        
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