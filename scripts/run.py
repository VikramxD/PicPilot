import argparse
import os
from mask_generator import generate_mask, invert_mask
from models import kandinsky_inpainting_inference, load_image
from PIL import Image

def main(args):
    # Generate mask
    mask = generate_mask(args.image_path)
    mask_image = Image.fromarray(mask)

    # Save original mask
    original_mask_path = os.path.join(args.mask_dir, f'{args.uid}_original_mask.jpg')
    mask_image.save(original_mask_path)
    
    
    # Invert mask
    mask_image = load_image(original_mask_path)
    inverted_mask = invert_mask(mask_image)
    inverted_mask_path = os.path.join(args.mask_dir, f'{args.uid}_inverted_mask.jpg')
    inverted_mask.save(inverted_mask_path)

    # Load mask and image
    invert_mask_image = load_image(inverted_mask_path)
    image = load_image(args.image_path)

    # Perform inpainting
    output_image = kandinsky_inpainting_inference(args.prompt, args.negative_prompt, image, mask_image)

    # Save output image
    output_image_path = os.path.join(args.output_dir, f'{args.uid}_output.jpg')
    output_image.save(output_image_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Kandinsky inpainting on an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the Kandinsky inpainting.')
    parser.add_argument('--negative_prompt', type=str, required=True, help='Negative prompt for the Kandinsky inpainting.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output image.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory to save the mask image.')
    parser.add_argument('--uid', type=str, required=True, help='Unique identifier for the image and mask.')

    args = parser.parse_args()
    main(args)