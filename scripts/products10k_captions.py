from datasets import load_dataset
from config import PRODUCTS_10k_DATASET, CAPTIONING_MODEL_NAME
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageCaptioner:
    def __init__(self, dataset: str, processor: str, model: str):
        self.dataset = load_dataset(dataset, split="train")
        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model).to(device)

    def process_dataset(self):
        # Assuming 'pixel_values' is the column name for images in the dataset
        self.dataset = self.dataset.rename_column("pixel_values", "image")
        # Remove unwanted columns
        if "label" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(["label"])
        return self.dataset

    def generate_captions(self):
        self.dataset = self.process_dataset()

        for idx in tqdm(range(len(self.dataset))):
            image = self.dataset[idx]["image"].convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model.generate(**inputs)
            blip_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            self.dataset[idx]["caption"] = blip_caption
            print(f"Caption for image {idx}: {blip_caption}")
        
        # Optionally, you can save the dataset with captions to disk
        # self.dataset.save_to_disk('path_to_save_dataset')
        
        return self.dataset

ic = ImageCaptioner(
    dataset=PRODUCTS_10k_DATASET,
    processor=CAPTIONING_MODEL_NAME,
    model=CAPTIONING_MODEL_NAME,
)
ic.generate_captions()
