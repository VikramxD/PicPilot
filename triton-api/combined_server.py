import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from PIL import Image
import io
from scripts.outpainting import ControlNetZoeDepthOutpainting

class OutpaintingModel:
    def __init__(self):
        self.processor = ControlNetZoeDepthOutpainting(target_size=(1024, 1024))

    @batch
    def outpaint(self, image, controlnet_prompt, controlnet_negative_prompt, controlnet_conditioning_scale,
                 controlnet_guidance_scale, controlnet_num_inference_steps, controlnet_guidance_end,
                 inpainting_prompt, inpainting_negative_prompt, inpainting_guidance_scale,
                 inpainting_strength, inpainting_num_inference_steps):
        results = []
        for img, *params in zip(image, controlnet_prompt, controlnet_negative_prompt, controlnet_conditioning_scale,
                                controlnet_guidance_scale, controlnet_num_inference_steps, controlnet_guidance_end,
                                inpainting_prompt, inpainting_negative_prompt, inpainting_guidance_scale,
                                inpainting_strength, inpainting_num_inference_steps):
            pil_image = Image.open(io.BytesIO(img.tobytes())).convert("RGB")
            result = self.processor.run_pipeline(
                pil_image, *params
            )
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            results.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
        return {"output_image": results}

model = OutpaintingModel()

triton = Triton()
triton.bind(
    model_name="outpainting",
    infer_func=model.outpaint,
    inputs=[
        Tensor(name="image", dtype=np.uint8, shape=(-1,)),
        Tensor(name="controlnet_prompt", dtype=np.object_, shape=(-1,)),
        Tensor(name="controlnet_negative_prompt", dtype=np.object_, shape=(-1,)),
        Tensor(name="controlnet_conditioning_scale", dtype=np.float32, shape=(-1,)),
        Tensor(name="controlnet_guidance_scale", dtype=np.float32, shape=(-1,)),
        Tensor(name="controlnet_num_inference_steps", dtype=np.int32, shape=(-1,)),
        Tensor(name="controlnet_guidance_end", dtype=np.float32, shape=(-1,)),
        Tensor(name="inpainting_prompt", dtype=np.object_, shape=(-1,)),
        Tensor(name="inpainting_negative_prompt", dtype=np.object_, shape=(-1,)),
        Tensor(name="inpainting_guidance_scale", dtype=np.float32, shape=(-1,)),
        Tensor(name="inpainting_strength", dtype=np.float32, shape=(-1,)),
        Tensor(name="inpainting_num_inference_steps", dtype=np.int32, shape=(-1,))
    ],
    outputs=[
        Tensor(name="output_image", dtype=np.uint8, shape=(-1,))
    ],
    config=ModelConfig(max_batch_size=4)
)

triton.serve()
