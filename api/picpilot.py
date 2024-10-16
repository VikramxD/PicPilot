from litserve.server import LitServer,run_all
from flux_serve import FluxInpaintingAPI
from sdxl_serve import SDXLLoraAPI
from configs.tti_settings import tti_settings
from outpainting_serve import OutpaintingAPI
from image2video_serve import ImageToVideoAPI
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*", # Allow all origins
]

flux_server = LitServer(FluxInpaintingAPI(), api_path='/api/v2/painting/flux', accelerator="auto",devices='auto', max_batch_size=4, batch_timeout=0.1,middlewares=((CORSMiddleware, {"allow_origins": origins, "allow_credentials": True, "allow_methods": ["*"], "allow_headers": ["*"],}),))
sdxl_server = LitServer(SDXLLoraAPI(), api_path='/api/v2/generate/sdxl', accelerator="auto",devices='auto', max_batch_size=tti_settings.MAX_BATCH_SIZE, batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6, middlewares=((CORSMiddleware, {"allow_origins": origins, "allow_credentials": True, "allow_methods": ["*"], "allow_headers": ["*"],}),))
outpainting_server = LitServer(OutpaintingAPI(), api_path='/api/v2/painting/sdxl_outpainting', accelerator='auto',devices='auto', max_batch_size=4, batch_timeout=0.1,middlewares=((CORSMiddleware, {"allow_origins": origins, "allow_credentials": True, "allow_methods": ["*"], "allow_headers": ["*"],}),))
image2video_server = LitServer(ImageToVideoAPI(), api_path='/api/v2/image2video/cogvideox', accelerator='auto',devices='auto', max_batch_size=1, batch_timeout=0.1,middlewares=((CORSMiddleware, {"allow_origins": origins, "allow_credentials": True, "allow_methods": ["*"], "allow_headers": ["*"],}),))


if __name__ == '__main__':
    run_all([flux_server,sdxl_server,outpainting_server,image2video_server], port=8000)