import sys
import torch
sys.path.append('../')
from Kandinsky.kandinsky3 import get_inpainting_pipeline
from utils import ImageAugmentation
from diffusers.utils import load_image
from PIL import Image

device_map = torch.device('cuda:0')
dtype_map = {
    'unet': torch.float16,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}


pipe = get_inpainting_pipeline(
    device_map, dtype_map,
)

augmenter = ImageAugmentation(target_width=2560, target_height=1440)
image = Image.open(image_path='/home/product_diffusion_api/sample_data/example1.jpg')
extended_image = augmenter.extend_image(image)
mask_image = augmenter.generate_mask_from_bbox(extended_image, segmentation_model='facebook/sam-vit-base', detection_model='yolov8s')
mask_image = augmenter.invert_mask(mask_image)


image = pipe( "Product on the Kitchen used for cooking", extended_image, mask_image)
image.save('output.jpg')
