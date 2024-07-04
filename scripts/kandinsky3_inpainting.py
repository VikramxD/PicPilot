import torch
from Kandinsky.kandinsky3 import get_inpainting_pipeline
from scripts.api_utils import ImageAugmentation,accelerator
from diffusers.utils import load_image
import numpy as np
from PIL import Image

device_map = torch.device(accelerator())
dtype_map = {
    'unet': torch.float16,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}


pipe = get_inpainting_pipeline(
    device_map, dtype_map,
)

image = Image.open('/home/PicPilot/sample_data/image.png')
mask_image = Image.open('/home/PicPilot/sample_data/mask_image.png')
image = load_image(image=image)
mask_image = np.array(mask_image)




image = pipe( "Product on the Kitchen used for cooking", image, mask_image)
image.save('output.jpg')
