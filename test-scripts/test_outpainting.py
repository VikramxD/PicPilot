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

def test_outpainting_api(server_url, input_image_path, prompt):
    """
    Test the Outpainting API by sending a request and processing the response.

    Args:
        server_url (str): URL of the Outpainting API server.
        input_image_path (str): Path to the input image file.
        prompt (str): The prompt for outpainting.

    Returns:
        None
    """
    # Prepare the request payload
    payload = {
        "image": encode_image_to_base64(input_image_path),
        "width": 2560,
        "height": 1440,
        "overlap_percentage": 10,
        "num_inference_steps": 8,
        "resize_option": "Full",
        "custom_resize_percentage": 100,
        "prompt_input": prompt,
        "alignment": "Middle",
        "overlap_left": True,
        "overlap_right": True,
        "overlap_top": True,
        "overlap_bottom": True
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
        print(f"Completion time: {result['completion_time']} seconds")
        print(f"Prompt ratio: {result['prompt_ratio']}")
        print(f"Image resolution: {result['image_resolution']}")

        # Decode and save the result image
        image_data = base64.b64decode(result['result'])
        result_image = Image.open(io.BytesIO(image_data))
        result_image.save("outpainting_result.png")
        print("Result image saved as 'outpainting_result.png'")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing response: {e}")
        print(f"Response content: {response.text}")

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000/predict"  # Adjust this to your server's address
    INPUT_IMAGE_PATH = "/root/PicPilot/sample_data/example3.jpg"  # Replace with your input image path
    PROMPT = "A beautiful landscape "  # Replace with your desired prompt

    test_outpainting_api(SERVER_URL, INPUT_IMAGE_PATH, PROMPT)