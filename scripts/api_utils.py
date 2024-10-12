"""
This module provides utilities for device management, memory handling, and file operations.
It includes functions for accelerator selection, memory clearing, and image/video processing.
"""

import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import List, Optional, Union
import gc
import io
from io import BytesIO
import base64
import uuid
import torch
from PIL import Image
from scripts.s3_manager import S3ManagerService

# Device Management
class DeviceManager:
    """
    Manages device selection for accelerated computing.

    This class handles the selection of appropriate computing devices (CPU, CUDA, MPS)
    based on system availability and user preferences.

    Attributes:
        _accelerator (str): The type of accelerator being used.
        _devices (Union[List[int], int]): The devices available for use.
    """

    def __init__(self, accelerator: str = "auto", devices: Union[List[int], int, str] = "auto"):
        """
        Initialize the DeviceManager.

        Args:
            accelerator (str): The type of accelerator to use. Defaults to "auto".
            devices (Union[List[int], int, str]): The devices to use. Defaults to "auto".
        """
        self._accelerator = self._sanitize_accelerator(accelerator)
        self._devices = self._setup_devices(devices)

    @property
    def accelerator(self):
        """Get the current accelerator type."""
        return self._accelerator

    @property
    def devices(self):
        """Get the current devices in use."""
        return self._devices

    @staticmethod
    def _sanitize_accelerator(accelerator: Optional[str]):
        """Sanitize the accelerator input."""
        if isinstance(accelerator, str):
            accelerator = accelerator.lower()
        if accelerator not in ["auto", "cpu", "mps", "cuda", "gpu", None]:
            raise ValueError("accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', or 'gpu'")
        return "auto" if accelerator is None else accelerator

    def _setup_devices(self, devices: Union[List[int], int, str]):
        """Set up the devices based on input and availability."""
        if devices == "auto":
            return self._auto_device_count()
        elif isinstance(devices, int):
            return min(devices, self._auto_device_count())
        elif isinstance(devices, list):
            return [dev for dev in devices if dev < self._auto_device_count()]
        else:
            raise ValueError("devices must be 'auto', an integer, or a list of integers")

    def _auto_device_count(self) -> int:
        """Automatically determine the number of available devices."""
        if self._accelerator == "cuda":
            return check_cuda_with_nvidia_smi()
        elif self._accelerator == "mps":
            return 1 
        elif self._accelerator == "cpu":
            return os.cpu_count() or 1
        else:
            return 1

    def _choose_auto_accelerator(self):
        """Choose the best available accelerator automatically."""
        gpu_backend = self._choose_gpu_accelerator_backend()
        return gpu_backend if gpu_backend else "cpu"

    @staticmethod
    def _choose_gpu_accelerator_backend():
        """Choose the appropriate GPU backend if available."""
        if check_cuda_with_nvidia_smi() > 0:
            return "cuda"
        if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
            return "mps"
        return None

@lru_cache(maxsize=1)
def check_cuda_with_nvidia_smi() -> int:
    """
    Check CUDA availability using nvidia-smi.

    Returns:
        int: The number of available CUDA devices.
    """
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        devices = [el for el in nvidia_smi_output.split("\n") if el.startswith("GPU")]
        devices = [el.split(":")[0].split()[1] for el in devices]
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            devices = [el for el in devices if el in visible_devices.split(",")]
        return len(devices)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0

def accelerator(devices: Union[List[int], int, str] = "auto") -> tuple:
    """
    Determine the device accelerator to use based on availability.

    Args:
        devices (Union[List[int], int, str]): Specifies the devices to use.

    Returns:
        tuple: A tuple containing the accelerator type and available devices.
    """
    device_manager = DeviceManager(accelerator="auto", devices=devices)
    return device_manager.accelerator, device_manager.devices

# Memory Management
def clear_memory():
    """
    Clear memory by collecting garbage and emptying the CUDA cache.
    """
    gc.collect()
    torch.cuda.empty_cache()

# File Operations
def pil_to_b64_json(image: Image.Image) -> dict:
    """
    Convert a PIL image to a base64-encoded JSON object.

    Args:
        image (PIL.Image.Image): The PIL image object to be converted.

    Returns:
        dict: A dictionary containing the image ID and the base64-encoded image.
    """
    image_id = str(uuid.uuid4())
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_id": image_id, "b64_image": b64_image}

def pil_to_s3_json(image: Image.Image, file_name: str) -> dict:
    """
    Upload a PIL image to Amazon S3 and return a JSON object with the image ID and signed URL.

    Args:
        image (PIL.Image.Image): The PIL image to be uploaded.
        file_name (str): The name of the file.

    Returns:
        dict: A JSON object containing the image ID and the signed URL.
    """
    image_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(image_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(unique_file_name, exp=43200)  # 12 hours
    return {"image_id": image_id, "url": signed_url}

def mp4_to_s3_json(video_bytes: io.BytesIO, file_name: str) -> dict:
    """
    Upload an MP4 video to Amazon S3 and return a JSON object with the video ID and signed URL.

    Args:
        video_bytes (io.BytesIO): The video data as bytes.
        file_name (str): The name of the file.

    Returns:
        dict: A JSON object containing the video ID and the signed URL.
    """
    video_id = str(uuid.uuid4())
    s3_uploader = S3ManagerService()

    unique_file_name = s3_uploader.generate_unique_file_name(file_name)
    s3_uploader.upload_file(video_bytes, unique_file_name)
    signed_url = s3_uploader.generate_signed_url(unique_file_name, exp=43200)  # 12 hours
    return {"video_id": video_id, "url": signed_url}

if __name__ == "__main__":
    acc, devs = accelerator()
    print(f"Selected accelerator: {acc}")
    print(f"Available devices: {devs}")