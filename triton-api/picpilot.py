from litserve.server import LitServer, run_all
from flux_serve import FluxInpaintingAPI
from sdxl_serve import SDXLLoraAPI
from configs.tti_settings import tti_settings


if __name__=="__main__":
    flux_server = LitServer(FluxInpaintingAPI(),api_path='/api/v2/inpainting/flux',accelerator="auto",max_batch_size=4,batch_timeout=0.1)
    sdxl_server = LitServer(SDXLLoraAPI(),api_path='/api/v2/generate/sdxl',accelerator="auto",max_batch_size=tti_settings.MAX_BATCH_SIZE,batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6)
    run_all([flux_server,sdxl_server], port=8000)
    
   