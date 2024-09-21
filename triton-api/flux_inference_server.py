import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from PIL import Image
import io
from typing import Dict
import logging
import json
from scripts.s3_manager import S3ManagerService
from config_settings import settings
from scripts.flux_inference import FluxInpaintingInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s3_manager = S3ManagerService()

# Initialize the FluxInpaintingInference
flux_inpainter = FluxInpaintingInference()

def decode_image(filename: np.ndarray) -> Image.Image:
    """
    Decode image from S3 and return as a PIL Image.

    Args:
        filename (np.ndarray): The filename of the image in S3 as a numpy array.

    Returns:
        Image.Image: The image data as a PIL Image.

    Raises:
        ValueError: If the image cannot be retrieved or decoded.
    """
    try:
        filename_str = filename.item().decode('utf-8') if isinstance(filename.item(), bytes) else filename.item()
        filename_str = filename_str.strip("b'\"")

        s3_object = s3_manager.get_object(filename_str, settings.AWS_BUCKET_NAME)
        if s3_object is None:
            raise ValueError(f"Failed to retrieve file {filename_str} from S3")
        
        return Image.open(io.BytesIO(s3_object["Body"].read()))
    except Exception as e:
        logger.error(f"Error in decode_image: {e}")
        raise

@batch
def _infer_fn(
    prompt: np.ndarray,
    image_filename: np.ndarray,
    mask_filename: np.ndarray,
    strength: np.ndarray,
    seed: np.ndarray,
    num_inference_steps: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Batch inference function for processing multiple requests.

    Args:
        prompt (np.ndarray): Array of prompts for image generation.
        image_filename (np.ndarray): Array of image filenames.
        mask_filename (np.ndarray): Array of mask filenames.
        strength (np.ndarray): Array of strengths for inpainting.
        seed (np.ndarray): Array of seeds for reproducibility.
        num_inference_steps (np.ndarray): Array of number of inference steps.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the 'output' key containing the batch results.
    """
    prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]

    outputs = []
    for p, img_filename, m_filename, s, sd, steps in zip(prompts, image_filename, mask_filename, strength, seed, num_inference_steps):
        try:
            input_image = decode_image(img_filename)
            mask_image = decode_image(m_filename)

            # Use FluxInpaintingInference to generate the inpainted image
            result = flux_inpainter.generate_inpainting(
                input_image=input_image,
                mask_image=mask_image,
                prompt=p,
                seed=sd.item(),
                strength=s.item(),
                num_inference_steps=steps.item()
            )

           
            output_image_np = np.array(result)
            
            # Convert the numpy array to a list and then to a JSON string
            output_json = json.dumps(output_image_np.tolist())
            outputs.append([output_json.encode()])
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            outputs.append([str(e).encode()])

    return {"output": np.array(outputs, dtype=object)}

def main():
    triton_config = TritonConfig(http_port=8000, metrics_port=8002, exit_on_error=True)

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="FLUX_INPAINTING_SERVER",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="image_filename", dtype=np.bytes_, shape=(1,)),
                Tensor(name="mask_filename", dtype=np.bytes_, shape=(1,)),
                Tensor(name="strength", dtype=np.float32, shape=(1,)),
                Tensor(name="seed", dtype=np.int32, shape=(1,)),
                Tensor(name="num_inference_steps", dtype=np.int32, shape=(1,)),
            ],
            outputs=[Tensor(name="output", dtype=np.bytes_, shape=(1,))],
            config=ModelConfig(
                max_batch_size=8,
                batcher=DynamicBatcher(max_queue_delay_microseconds=10000),
            ),
        )
        triton.serve()

if __name__ == "__main__":
    main()