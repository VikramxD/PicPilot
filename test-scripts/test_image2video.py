import requests
import base64
import json
from PIL import Image
import io
import os

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def validate_image(base64_string):
    """
    Validate the base64 encoded image by attempting to open it with PIL.

    Args:
        base64_string (str): Base64 encoded image string.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        image_data = base64.b64decode(base64_string)
        Image.open(io.BytesIO(image_data))
        return True
    except Exception as e:
        print(f"Error validating image: {e}")
        return False

def test_image_to_video_api(server_url, input_image_path, prompt):
    """
    Test the Image-to-Video API by sending a request and processing the response.

    Args:
        server_url (str): URL of the Image-to-Video API server.
        input_image_path (str): Path to the input image file.
        prompt (str): The prompt for video generation.

    Returns:
        None
    """
    # Encode and validate the image
    base64_image = encode_image_to_base64(input_image_path)
    if not base64_image:
        print("Failed to encode image.")
        return
    
    if not validate_image(base64_image):
        print("Encoded image is not valid.")
        return

    # Prepare the request payload
    payload = {
        "image": base64_image,
        "prompt": prompt,
        "num_frames": 49,
        "num_inference_steps": 20,
        "guidance_scale": 6.0,
        "height": 480,
        "width": 720,
        "use_dynamic_cfg": True,
        "fps": 10
    }

    # Send POST request to the server
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return

    # Process the response
    try:
        result = response.json()
        print(f"Response received:")
        print(f"Completion time: {result['completion_time']} seconds")
        print(f"Video resolution: {result['video_resolution']}")
        print(f"FPS: {result['fps']}")

        # Save the result video
        video_url = result['result']['url']
        video_response = requests.get(video_url)
        video_response.raise_for_status()

        output_path = "generated_video.mp4"
        with open(output_path, "wb") as video_file:
            video_file.write(video_response.content)
        
        print(f"Result video saved as '{output_path}'")

    except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException) as e:
        print(f"Error processing response: {e}")
        print(f"Response content: {response.text}")

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000/predict"  # Adjust this to your server's address
    INPUT_IMAGE_PATH = "/root/PicPilot/sample_data/product_img.jpg"  # Replace with your input image path
    PROMPT = "A product shot of a Nike Shoe in a studio High quality, ultrarealistic detail and breath-taking movie-like camera shot"  # Replace with your desired prompt

    test_image_to_video_api(SERVER_URL, INPUT_IMAGE_PATH, PROMPT)