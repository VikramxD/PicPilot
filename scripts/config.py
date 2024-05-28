<<<<<<< HEAD
<<<<<<< HEAD
MODEL_NAME:str="stabilityai/stable-diffusion-xl-base-1.0"
ADAPTER_NAME:str = "VikramSingh178/sdxl-lora-finetune-product-caption"
ADAPTER_NAME_2:str = "VikramSingh178/Products10k-SDXL-Lora"
VAE_NAME:str= "madebyollin/sdxl-vae-fp16-fix"
DATASET_NAME:str = "hahminlew/kream-product-blip-captions"
PROJECT_NAME:str = "Product Photography"
PRODUCTS_10k_DATASET:str = "VikramSingh178/Products-10k-BLIP-captions"
CAPTIONING_MODEL_NAME:str = "Salesforce/blip-image-captioning-base"
SEGMENTATION_MODEL_NAME:str = "facebook/sam-vit-large"
DETECTION_MODEL_NAME:str = "yolov8l"
ENABLE_COMPILE:bool = True
INPAINTING_MODEL_NAME:str = 'kandinsky-community/kandinsky-2-2-decoder-inpaint'


=======
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
ADAPTER_NAME = "VikramSingh178/sdxl-lora-finetune-product-caption"
ADAPTER_NAME_2 = "VikramSingh178/Products10k-SDXL-Lora"
VAE_NAME= "madebyollin/sdxl-vae-fp16-fix"
DATASET_NAME= "hahminlew/kream-product-blip-captions"
PROJECT_NAME = "Product Photography"
PRODUCTS_10k_DATASET = "VikramSingh178/Products-10k-BLIP-captions"
CAPTIONING_MODEL_NAME = "Salesforce/blip-image-captioning-base"
SEGMENTATION_MODEL_NAME = "facebook/sam-vit-large"
DETECTION_MODEL_NAME = "yolov8l"
>>>>>>> a817fb6 (chore: Update .gitignore and add new files for inpainting pipeline)



class Config:
    def __init__(self):
        self.pretrained_model_name_or_path = MODEL_NAME
        self.pretrained_vae_model_name_or_path = VAE_NAME
        self.revision = None
        self.variant = None
        self.dataset_name = PRODUCTS_10k_DATASET
        self.dataset_config_name = None
        self.train_data_dir = None
        self.image_column = 'image'
        self.caption_column = 'text'
        self.validation_prompt = None
        self.num_validation_images = 4
        self.validation_epochs = 1
        self.max_train_samples = 7
        self.output_dir = "output"
        self.cache_dir = None
        self.seed = 42
        self.resolution = 512
        self.center_crop = True
        self.random_flip = True
        self.train_text_encoder = False
        self.train_batch_size = 64
        self.num_train_epochs = 400
        self.max_train_steps = None
        self.checkpointing_steps = 500
        self.checkpoints_total_limit = None
        self.resume_from_checkpoint = None
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = False
        self.learning_rate = 1e-4
        self.scale_lr = False
        self.lr_scheduler = "constant"
        self.lr_warmup_steps = 500
        self.snr_gamma = None
        self.allow_tf32 = True
        self.dataloader_num_workers = 0
        self.use_8bit_adam = True
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        self.push_to_hub = True
        self.hub_token = None
        self.prediction_type = None
        self.hub_model_id = None
        self.logging_dir = "logs"
        self.report_to = "wandb"
        self.mixed_precision = 'fp16'
        self.local_rank = -1
        self.enable_xformers_memory_efficient_attention = False
        self.noise_offset = 0
        self.rank = 4
        self.debug_loss = False


=======
LOGS_DIR = '../logs'
DATA_DIR = '../data'
Project_Name = 'product_placement_api'
entity = 'vikramxd'
image_dir = '../sample_data'
mask_dir = '../masks'
segmentation_model = 'facebook/sam-vit-large'
detection_model = 'yolov8l'
kandinsky_model_name = 'kandinsky-community/kandinsky-2-2-decoder-inpaint'
video_model_name = 'stabilityai/stable-video-diffusion-img2vid-xt'
target_width = 2560
target_height = 1440
roi_scale = 0.6
>>>>>>> aaed2f5 (Refactor config.py and models.py, and add new functions in segment_everything.py and video_pipeline.py)
