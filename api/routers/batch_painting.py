import sys
sys.path.append('../scripts')
import os
import uuid
from typing import List, Tuple, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image
import lightning.pytorch as pl
from utils import pil_to_s3_json, pil_to_b64_json, ImageAugmentation, accelerator
from inpainting_pipeline import load_pipeline
from async_batcher.batcher import AsyncBatcher
from config import INPAINTING_MODEL_NAME
router = APIRouter()


inpainting_pipeline = load_pipeline(model_name = INPAINTING_MODEL_NAME,enable_)
