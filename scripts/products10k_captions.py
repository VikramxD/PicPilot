import torch
from datasets import load_dataset, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# Assuming PRODUCTS_10k_DATASET and CAPTIONING_MODEL_NAME are defined in config.py
from config import PRODUCTS_10k_DATASET, CAPTIONING_MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageCaptioner:
    """
    A class for generating captions for images using a pre-trained model.

    Args:
        dataset (str): The path to the dataset.
        processor (str): The pre-trained processor model to use for image processing.
        model (str): The pre-trained model to use for caption generation.
        prompt (str): The conditioning prompt to use for caption generation.

    Attributes:
        dataset: The loaded dataset.
        processor: The pre-trained processor model.
        model: The pre-trained caption generation model.
        prompt: The conditioning prompt for generating captions.

    Methods:
        process_dataset: Preprocesses the dataset.
        generate_caption: Generates a caption for a single image.
        generate_captions: Generates captions for all images in the dataset.
    """

    def __init__(self, dataset: str, processor: str, model: str, prompt: str = "Product photo of"):
        self.dataset = load_dataset(dataset, split="test")
        self.dataset = self.dataset.select(range(10000))  # For demonstration purposes
        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model).to(device)
        self.prompt = prompt

    def process_dataset(self):
        """
        Preprocesses the dataset by renaming the image column and removing unwanted columns.

        Returns:
            The preprocessed dataset.
        """
        # Check if 'image' column exists, otherwise use 'pixel_values' if it exists
        image_column = "image" if "image" in self.dataset.column_names else "pixel_values"
        self.dataset = self.dataset.rename_column(image_column, "image")

        if "label" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(["label"])

        # Add an empty 'text' column for captions if it doesn't exist
        if "text" not in self.dataset.column_names:
            new_column = [""] * len(self.dataset)
            self.dataset = self.dataset.add_column("text", new_column)

        return self.dataset

    def generate_caption(self, example):
        """
        Generates a caption for a single image.

        Args:
            example (dict): A dictionary containing the image data.

        Returns:
            dict: The dictionary with the generated caption.
        """
        image = example["image"].convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        prompt_inputs = self.processor(text=[self.prompt], return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, **prompt_inputs)
        blip_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        example["text"] = blip_caption
        return example

    def generate_captions(self):
        """
        Generates captions for all images in the dataset.

        Returns:
            Dataset: The dataset with generated captions.
        """
        self.dataset = self.process_dataset()
        self.dataset = self.dataset.map(self.generate_caption, batched=False)
        return self.dataset

# Initialize ImageCaptioner
ic = ImageCaptioner(
    dataset=PRODUCTS_10k_DATASET,
    processor=CAPTIONING_MODEL_NAME,
    model=CAPTIONING_MODEL_NAME,
    prompt='Commercial photography of'
)

# Generate captions for the dataset
products10k_dataset = ic.generate_captions()

# Save the dataset to the hub
products10k_dataset.push_to_hub("VikramSingh178/Products-10k-BLIP-captions")
