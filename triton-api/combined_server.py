import numpy as np
from typing import List, Dict, Any
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from PIL import Image

from scripts.outpainting import ControlNetZoeDepthOutpainting
from scripts.api_utils import pil_to_b64_json

class OutpaintingModel:
    """
    A class to handle outpainting operations using ControlNetZoeDepthOutpainting.

    This class initializes the outpainting pipeline and provides methods to process
    outpainting requests.
    """

    def __init__(self, target_size: tuple = (1024, 1024)):
        """
        Initialize the OutpaintingModel with a specified target size.

        Args:
            target_size (tuple): The target size for the outpainted image. Defaults to (1024, 1024).
        """
        self.pipeline = ControlNetZoeDepthOutpainting(target_size=target_size)

    @staticmethod
    def _decode_string(value: np.ndarray) -> str:
        """
        Decode a numpy array of bytes to a UTF-8 string.

        Args:
            value (np.ndarray): A numpy array containing a byte string.

        Returns:
            str: The decoded UTF-8 string.
        """
        return value[0].decode('utf-8')

    @staticmethod
    def _get_float(value: np.ndarray) -> float:
        """
        Extract a float value from a numpy array.

        Args:
            value (np.ndarray): A numpy array containing a single float value.

        Returns:
            float: The extracted float value.
        """
        return float(value[0])

    @staticmethod
    def _get_int(value: np.ndarray) -> int:
        """
        Extract an integer value from a numpy array.

        Args:
            value (np.ndarray): A numpy array containing a single integer value.

        Returns:
            int: The extracted integer value.
        """
        return int(value[0])

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
        """
        Perform outpainting on the given images with the specified parameters.

        This method processes a batch of images, applying outpainting to each one
        based on the provided parameters.

        Args:
            image (np.ndarray): Batch of input images.
            base_image_description (np.ndarray): Descriptions of the base images.
            base_image_negative_prompt (np.ndarray): Negative prompts for the base images.
            controlnet_conditioning_scale (np.ndarray): ControlNet conditioning scales.
            controlnet_guidance_scale (np.ndarray): ControlNet guidance scales.
            controlnet_num_inference_steps (np.ndarray): Number of inference steps for ControlNet.
            controlnet_guidance_end (np.ndarray): ControlNet guidance end values.
            background_extension_prompt (np.ndarray): Prompts for background extension.
            outpainting_negative_prompt (np.ndarray): Negative prompts for outpainting.
            outpainting_guidance_scale (np.ndarray): Guidance scales for outpainting.
            outpainting_strength (np.ndarray): Strengths for outpainting.
            outpainting_num_inference_steps (np.ndarray): Number of inference steps for outpainting.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the results of the outpainting process.
        """
        results = []
        for i in range(len(image)):
            pil_image = Image.fromarray(image[i].astype('uint8'))
            
            result = self.pipeline.run_pipeline(
                pil_image,
                base_image_description=self._decode_string(base_image_description[i]),
                base_image_negative_prompt=self._decode_string(base_image_negative_prompt[i]),
                controlnet_conditioning_scale=self._get_float(controlnet_conditioning_scale[i]),
                controlnet_guidance_scale=self._get_float(controlnet_guidance_scale[i]),
                controlnet_num_inference_steps=self._get_int(controlnet_num_inference_steps[i]),
                controlnet_guidance_end=self._get_float(controlnet_guidance_end[i]),
                background_extension_prompt=self._decode_string(background_extension_prompt[i]),
                outpainting_negative_prompt=self._decode_string(outpainting_negative_prompt[i]),
                outpainting_guidance_scale=self._get_float(outpainting_guidance_scale[i]),
                outpainting_strength=self._get_float(outpainting_strength[i]),
                outpainting_num_inference_steps=self._get_int(outpainting_num_inference_steps[i])
            )
            
            result_json = pil_to_b64_json(result)
            results.append(result_json.encode('utf-8'))
        
        return {"result": np.array(results)}

def create_input_tensor(name: str, dtype: np.dtype, shape: tuple) -> Tensor:
    """
    Create an input tensor for the PyTriton model.

    Args:
        name (str): The name of the tensor.
        dtype (np.dtype): The data type of the tensor.
        shape (tuple): The shape of the tensor.

    Returns:
        Tensor: A PyTriton Tensor object.
    """
    return Tensor(name=name, dtype=dtype, shape=shape)

def setup_triton_model(model: OutpaintingModel) -> None:
    """
    Set up and serve the PyTriton model.

    This function binds the outpainting model to PyTriton and starts serving it.

    Args:
        model (OutpaintingModel): An instance of the OutpaintingModel class.
    """
    triton = Triton()
    triton.bind(
        model_name="outpainting",
        infer_func=model.outpaint,
        inputs=[
            create_input_tensor("image", np.uint8, (-1,)),
            create_input_tensor("base_image_description", np.bytes_, (-1, 1)),
            create_input_tensor("base_image_negative_prompt", np.bytes_, (-1, 1)),
            create_input_tensor("controlnet_conditioning_scale", np.float32, (-1, 1)),
            create_input_tensor("controlnet_guidance_scale", np.float32, (-1, 1)),
            create_input_tensor("controlnet_num_inference_steps", np.int32, (-1, 1)),
            create_input_tensor("controlnet_guidance_end", np.float32, (-1, 1)),
            create_input_tensor("background_extension_prompt", np.bytes_, (-1, 1)),
            create_input_tensor("outpainting_negative_prompt", np.bytes_, (-1, 1)),
            create_input_tensor("outpainting_guidance_scale", np.float32, (-1, 1)),
            create_input_tensor("outpainting_strength", np.float32, (-1, 1)),
            create_input_tensor("outpainting_num_inference_steps", np.int32, (-1, 1))
        ],
        outputs=[
            create_input_tensor("result", np.bytes_, (-1,))
        ],
        config=ModelConfig(max_batch_size=8)
    )
    triton.serve()

if __name__ == "__main__":
    model = OutpaintingModel()
    setup_triton_model(model)
