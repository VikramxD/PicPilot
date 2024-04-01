import argparse
import os
from segment_everything import generate_mask_from_bbox, extend_image, invert_mask
from models import kandinsky_inpainting_inference, load_image
from PIL import Image
from config import segmentation_model, detection_model,target_height, target_width, roi_scale

def main(args):
    """
    Main function that performs the product diffusion process.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        None
    """
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    output_image_path = os.path.join(args.output_dir, f'{args.uid}_output.jpg')
    image = load_image(args.image_path)
    extended_image = extend_image(image, target_width=target_width, target_height=target_height, roi_scale=roi_scale)
    mask = generate_mask_from_bbox(extended_image, segmentation_model, detection_model)
    mask_image = Image.fromarray(mask)
    inverted_mask = invert_mask(mask_image)
    #inverted_mask = Image.fromarray(inverted_mask)
    output_image = kandinsky_inpainting_inference(args.prompt, args.negative_prompt, extended_image, inverted_mask)
    output_image.save(output_image_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Outpainting on an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the Kandinsky inpainting.')
    parser.add_argument('--negative_prompt', type=str, required=True, help='Negative prompt for the Kandinsky inpainting.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output image.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory to save the mask image.')
    parser.add_argument('--uid', type=str, required=True, help='Unique identifier for the image and mask.')
    args = parser.parse_args()
    main(args)