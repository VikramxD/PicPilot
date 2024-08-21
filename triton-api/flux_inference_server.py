import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from PIL import Image
import io
from typing import Dict
import logging
import requests
from scripts.s3_manager import S3ManagerService
from config_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s3_manager = S3ManagerService()


def decode_image(url: str, filename: np.ndarray) -> np.ndarray:
    """
    Decode image from S3 and convert to a normalized numpy array.

    Args:
        url (str): The URL of the image (not used in this implementation).
        filename (np.ndarray): The filename of the image in S3 as a numpy array.

    Returns:
        np.ndarray: The image data as a normalized numpy array in FP32 format.

    Raises:
        ValueError: If the image cannot be retrieved or decoded.
    """
    try:
        # Convert filename from numpy array to string
        filename_str = filename.item().decode('utf-8') if isinstance(filename.item(), bytes) else filename.item()
        
        # Remove any surrounding quotes or 'b' prefix
        filename_str = filename_str.strip("b'\"")

        s3_object = s3_manager.get_object(filename_str, settings.AWS_BUCKET_NAME)
        if s3_object is None:
            raise ValueError(f"Failed to retrieve file {filename_str} from S3")
        
        image = Image.open(io.BytesIO(s3_object["Body"].read()))
        image_np = np.array(image).astype(np.float32) / 255.0
        return image_np
    except Exception as e:
        logger.error(f"Error in decode_image: {e}")
        raise

@batch
def _infer_fn(
    prompt: np.ndarray,
    image_url: np.ndarray,
    image_filename: np.ndarray,
    mask_url: np.ndarray,
    mask_filename: np.ndarray,
    strength: np.ndarray,
    seed: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Batch inference function for processing multiple requests.

    Args:
        prompt (np.ndarray): Array of prompts for image generation.
        image_url (np.ndarray): Array of image URLs.
        image_filename (np.ndarray): Array of image filenames.
        mask_url (np.ndarray): Array of mask URLs.
        mask_filename (np.ndarray): Array of mask filenames.
        strength (np.ndarray): Array of strengths for inpainting.
        seed (np.ndarray): Array of seeds for reproducibility.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the 'output' key containing the batch results.
    """
    prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]

    masks = []
    images = []
    for m_filename, img_filename in zip(mask_filename, image_filename):
        try:
            masks.append(decode_image("", m_filename))
            images.append(decode_image("", img_filename))
        except ValueError as e:
            logger.error(f"Error decoding image: {e}")
            return {"error": np.array([str(e).encode()])}

    # Process each input
    outputs = []
    for p, m, img, s, sd in zip(prompts, masks, images, strength, seed):
        # Convert mask and image to torch tensors
        mask_tensor = torch.tensor(m).to(DEVICE).unsqueeze(0)
        image_tensor = torch.tensor(img).to(DEVICE).unsqueeze(0)

        # Model inference logic...
        # For demonstration, assume we output the image tensor directly
        output_image = image_tensor  # Replace with actual model inference logic

        output_image_np = output_image.squeeze().cpu().numpy()
        output_image_bytes = output_image_np.tobytes()
        outputs.append([output_image_bytes])

    return {"output": np.array(outputs)}

def main():
    triton_config = TritonConfig(http_port=8000, metrics_port=8002, exit_on_error=True)

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="FLUX_INPAINTING_SERVER",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="image_url", dtype=np.bytes_, shape=(1,)),
                Tensor(name="image_filename", dtype=np.bytes_, shape=(1,)),
                Tensor(name="mask_url", dtype=np.bytes_, shape=(1,)),
                Tensor(name="mask_filename", dtype=np.bytes_, shape=(1,)),
                Tensor(name="strength", dtype=np.float32, shape=(1,)),
                Tensor(name="seed", dtype=np.int32, shape=(1,)),
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
