import requests
import base64
import json
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_sdxl_lora_api(server_url, prompt, negative_prompt="", num_images=1, num_inference_steps=30, guidance_scale=7.5, mode="b64_json"):
    """
    Test the SDXL Lora API by sending a request and processing the response.

    Args:
        server_url (str): URL of the SDXL Lora API server.
        prompt (str): The prompt for image generation.
        negative_prompt (str, optional): The negative prompt for image generation. Defaults to "".
        num_images (int, optional): Number of images to generate. Defaults to 1.
        num_inference_steps (int, optional): Number of inference steps. Defaults to 30.
        guidance_scale (float, optional): Guidance scale for image generation. Defaults to 7.5.
        mode (str, optional): Response mode ('b64_json' or 's3_json'). Defaults to "b64_json".

    Returns:
        None
    """
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images": num_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "mode": mode
    }

    # Send POST request to the server
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return

    # Process the response
    try:
        result = response.json()
        print(f"Response received:")
        if mode == "s3_json":
            print(f"Result URL: {result['s3_url']}")
            print(f"Filename: {result['filename']}")
        elif mode == "b64_json":
            print("Image received in base64 format")
        
        print(f"Prompt: {prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Num Inference Steps: {num_inference_steps}")
        print(f"Guidance Scale: {guidance_scale}")

        # Save the result image
        if mode == "s3_json":
            image_response = requests.get(result['s3_url'])
            image_response.raise_for_status()
            result_image = Image.open(io.BytesIO(image_response.content))
        elif mode == "b64_json":
            image_data = base64.b64decode(result['b64_json'])
            result_image = Image.open(io.BytesIO(image_data))
        
        result_image.save("sdxl_result.png")
        print("Result image saved as 'sdxl_result.png'")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing response: {e}")
        print(f"Response content: {response.text}")

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000/predict"  # Adjust this to your server's address
    PROMPT = "A beautiful landscape with mountains and a lake"
    NEGATIVE_PROMPT = "ugly, blurry"
    NUM_IMAGES = 1
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    MODE = "b64_json"  # or "s3_json"

    test_sdxl_lora_api(SERVER_URL, PROMPT, NEGATIVE_PROMPT, NUM_IMAGES, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, MODE)