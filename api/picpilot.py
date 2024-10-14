import runpod
from litserve.server import LitServer
from flux_serve import FluxInpaintingAPI
from sdxl_serve import SDXLLoraAPI
from configs.tti_settings import tti_settings
from outpainting_serve import OutpaintingAPI
from image2video_serve import ImageToVideoAPI

# Initialize all servers
flux_server = LitServer(FluxInpaintingAPI(), api_path='/api/v2/painting/flux', accelerator="auto", max_batch_size=4, batch_timeout=0.1)
sdxl_server = LitServer(SDXLLoraAPI(), api_path='/api/v2/generate/sdxl', accelerator="auto", max_batch_size=tti_settings.MAX_BATCH_SIZE, batch_timeout=tti_settings.MAX_QUEUE_DELAY_MICROSECONDS / 1e6)
outpainting_server = LitServer(OutpaintingAPI(), api_path='/api/v2/painting/sdxl_outpainting', accelerator='auto', max_batch_size=4, batch_timeout=0.1)
image2video_server = LitServer(ImageToVideoAPI(), api_path='/api/v2/image2video/cogvideox', accelerator='auto', max_batch_size=1, batch_timeout=0.1)

# Map of API paths to their respective servers
server_map = {
    '/api/v2/painting/flux': flux_server,
    '/api/v2/generate/sdxl': sdxl_server,
    '/api/v2/painting/sdxl_outpainting': outpainting_server,
    '/api/v2/image2video/cogvideox': image2video_server
}

def picpilot_handler(job):
    job_input = job["input"]
    
    if "api_path" not in job_input or "data" not in job_input:
        return {"error": "Invalid input. Both 'api_path' and 'data' are required."}

    api_path = job_input["api_path"]
    data = job_input["data"]

    if api_path not in server_map:
        return {"error": f"Invalid API path: {api_path}"}

    server = server_map[api_path]
    
    try:
        # Process the request using the appropriate server
        result = server.process(data)
        return {"success": True, "result": result}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": picpilot_handler})