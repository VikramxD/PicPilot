from PIL import Image
import numpy as np
import PIL.ImageOps
from diffusers.utils import load_image






def invert_mask(mask_image: Image) -> np.ndarray:
    """Method to invert mask
    Args:
        mask_image (np.ndarray): input mask image
    Returns:
        np.ndarray: inverted mask image
    """
    inverted_mask_image =PIL.ImageOps.invert(mask_image)
    return inverted_mask_image





def extend_image(image_path, target_width, target_height, roi_scale=0.5):
    # Open the original image
    original_image = Image.open(image_path)

    # Get the dimensions of the original image
    original_width, original_height = original_image.size

    # Calculate the scale to fit the target resolution while keeping the aspect ratio
    scale = min(target_width / original_width, target_height / original_height)

    # Calculate the new dimensions of the image
    new_width = int(original_width * scale * roi_scale)
    new_height = int(original_height * scale * roi_scale)

    # Resize the original image with keeping the aspect ratio
    original_image_resized = original_image.resize((new_width, new_height))

    # Create a new image with white background
    extended_image = Image.new("RGB", (target_width, target_height), "white")

    # Calculate the position to paste the resized image at the center
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste the resized image onto the new image
    extended_image.paste(original_image_resized, (paste_x, paste_y))

    return extended_image







