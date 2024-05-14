from datasets import load_dataset
from config import PRODUCTS_10k_DATASET
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_dataset(PRODUCTS_10k_DATASET)


def image_captioning(processor , )






