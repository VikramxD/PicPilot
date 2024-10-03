import requests
import json
import base64
from PIL import Image
import io

class SDXLLoraClient:
    """
    Client for interacting with the SDXL LoRA server.
    """

    def __init__(self, base_url: str):
        """
        Initialize the client with the server's base URL.

        Args:
            base_url (str): The base URL of the SDXL LoRA server.
        """
        self.base_url = base_url

    def generate_image(self, prompt: str, negative_prompt: str = "", num_images: int = 1,
                       num_inference_steps: int = 50, guidance_scale: float = 7.5,
                       mode: str = "b64_json") -> list:
        """
        Send a request to the server to generate images.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str, optional): The negative prompt. Defaults to "".
            num_images (int, optional): Number of images to generate. Defaults to 1.
            num_inference_steps (int, optional): Number of inference steps. Defaults to 50.
            guidance_scale (float, optional): Guidance scale. Defaults to 7.5.
            mode (str, optional): Response mode ('b64_json' or 's3_json'). Defaults to "b64_json".

        Returns:
            list: A list of generated images (as PIL Image objects) or S3 URLs.
        """
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": num_images,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "mode": mode
        }

        response = requests.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()

        result = response.json()
        print(f"Server response: {result}")  # Debug print

        if isinstance(result, str):
            print(f"Unexpected string result: {result}")
            return [result]

        if mode == "b64_json":
            if isinstance(result, list):
                return [Image.open(io.BytesIO(base64.b64decode(img["base64"]))) for img in result]
            elif isinstance(result, dict) and "base64" in result:
                return [Image.open(io.BytesIO(base64.b64decode(result["base64"])))]
            else:
                raise ValueError(f"Unexpected result format for b64_json mode: {result}")
        elif mode == "s3_json":
            if isinstance(result, list):
                return [img["url"] for img in result if "url" in img]
            elif isinstance(result, dict) and "url" in result:
                return [result["url"]]
            else:
                raise ValueError(f"Unexpected result format for s3_json mode: {result}")
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

def main():
    """
    Main function to demonstrate the usage of the SDXLLoraClient.
    """
    client = SDXLLoraClient("http://localhost:8000")

    # Test case 1: Generate a single image
    print("Generating a single image...")
    images = client.generate_image(
        prompt="A serene landscape with mountains and a lake",
        negative_prompt='Low resolution , Poor Resolution',
        mode="s3_json"
    )

    # Test case 2: Generate multiple images
    print("\nGenerating multiple images...")
    images = client.generate_image(
        prompt="A futuristic cityscape at night",
        num_images=3,
        num_inference_steps=30,
        guidance_scale=8.0,
        mode="s3_json"
    )
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            img.save(f"test_image_2_{i+1}.png")
            print(f"Image saved as test_image_2_{i+1}.png")
        else:
            print(f"Unexpected result for image {i+1}: {img}")

    # Test case 3: Generate image with S3 storage
    print("\nGenerating image with S3 storage...")
    urls = client.generate_image(
        prompt="An abstract painting with vibrant colors",
        mode="s3_json"
    )
    print(f"S3 URLS for the generated image: {urls}")

if __name__ == "__main__":
    main()