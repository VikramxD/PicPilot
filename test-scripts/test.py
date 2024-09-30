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

def test_flux_inpainting_api(server_url, input_image_path, mask_image_path, prompt):
    """
    Test the Flux Inpainting API by sending a request and processing the response.

    Args:
        server_url (str): URL of the Flux Inpainting API server.
        input_image_path (str): Path to the input image file.
        mask_image_path (str): Path to the mask image file.
        prompt (str): The prompt for inpainting.

    Returns:
        None
    """
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "strength": 0.8,
        "seed": 42,
        "num_inference_steps": 50,
        "input_image": encode_image_to_base64(input_image_path),
        "mask_image": encode_image_to_base64(mask_image_path)
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
        print(f"Result URL: {result['result_url']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Seed: {result['seed']}")
        print(f"Time taken: {result['time_taken']} seconds")

        # Download and save the result image
        image_response = requests.get(result['result_url'])
        image_response.raise_for_status()
        result_image = Image.open(io.BytesIO(image_response.content))
        result_image.save("inpainting_result.png")
        print("Result image saved as 'inpainting_result.png'")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing response: {e}")
        print(f"Response content: {response.text}")

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000/api/v2/inpainting/flux"  # Adjust this to your server's address
    INPUT_IMAGE_PATH = "/root/PicPilot/sample_data/image.jpg"  # Replace with your input image path
    MASK_IMAGE_PATH = "/root/PicPilot/sample_data/mask.png"  # Replace with your mask image path
    PROMPT = "Signora Cooker"  # Replace with your desired prompt

    test_flux_inpainting_api(SERVER_URL, INPUT_IMAGE_PATH, MASK_IMAGE_PATH, PROMPT)