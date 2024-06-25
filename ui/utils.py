from scripts.api_utils import ImageAugmentation
from PIL import Image




def augment_image(image_path, target_width, target_height, roi_scale, segmentation_model_name, detection_model_name):
    """
    Augment an image by extending its dimensions and generating masks.

    Args:
        image_path (str): Path to the image file.
        target_width (int): Target width for augmentation.
        target_height (int): Target height for augmentation.
        roi_scale (float): Scale factor for region of interest.
        segmentation_model_name (str): Name of the segmentation model.
        detection_model_name (str): Name of the detection model.

    Returns:
        Tuple[Image.Image, Image.Image]: Augmented image and inverted mask.
    """
    image = Image.open(image_path)
    image_augmentation = ImageAugmentation(target_width, target_height, roi_scale)
    image = image_augmentation.extend_image(image)
    mask = image_augmentation.generate_mask_from_bbox(image, segmentation_model_name, detection_model_name)
    inverted_mask = image_augmentation.invert_mask(mask)
    return image, inverted_mask