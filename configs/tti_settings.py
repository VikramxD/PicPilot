from pydantic_settings import BaseSettings

class TTI_SETTINGS(BaseSettings):

    MODEL_NAME:str="stabilityai/stable-diffusion-xl-base-1.0"
    ADAPTER_NAME:str = "VikramSingh178/sdxl-lora-finetune-product-caption"
    ENABLE_COMPILE: bool = False
    DEVICE: str = "cuda"
    TRITON_MODEL_NAME: str = "PICPILOT_PRODUCTION_SERVER"
    MAX_BATCH_SIZE: int = 8
    MAX_QUEUE_DELAY_MICROSECONDS: int = 100
    TORCH_INDUCTOR_CONFIG: dict = {
        "conv_1x1_as_mm": True,
        "coordinate_descent_tuning": True,
        "epilogue_fusion": False,
        "coordinate_descent_check_all_directions": True,
        "force_fuse_int_mm_with_mul": True,
        "use_mixed_mm": True
    }
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    LOG_LEVEL: str = "INFO"

settings = TTI_SETTINGS()