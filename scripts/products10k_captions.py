from datasets import load_dataset, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from config import PRODUCTS_10k_DATASET,CAPTIONING_MODEL_NAME
import torch


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
        generate_captions: Generates captions for the images in the dataset.

    """

    def __init__(self, dataset: str, processor: str, model: str, prompt: str = "Product photo of"):
        self.dataset = load_dataset(dataset, split="train")
        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model).to(device)
        self.prompt = prompt

    def process_dataset(self):
        """
        Preprocesses the dataset by renaming the image column and removing unwanted columns.

        Returns:
            The preprocessed dataset.

        """
        self.dataset = self.dataset.rename_column("pixel_values", "image")
        if "label" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(["label"])
        return self.dataset

    def generate_captions(self):
        """
        Generates captions for the images in the dataset.

        Returns:
            The dataset with captions.

        """
        self.dataset = self.process_dataset()

        for idx in tqdm(range(len(self.dataset))):
            image = self.dataset[idx]["image"].convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            prompt_inputs = self.processor(text=[self.prompt], return_tensors="pt").to(device)
            outputs = self.model.generate(**inputs, **prompt_inputs)
            blip_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            self.dataset[idx]["text"] = blip_caption
      

       

        return self.dataset

# Initialize ImageCaptioner
ic = ImageCaptioner(
    dataset=PRODUCTS_10k_DATASET,
    processor=CAPTIONING_MODEL_NAME,
    model=CAPTIONING_MODEL_NAME,
    prompt='Photography of '  # Adding the conditioning prompt
)

# Generate captions for the dataset
products10k_dataset = ic.generate_captions()
new_dataset = Dataset.from_pandas(products10k_dataset.to_pandas())  # Convert to a `datasets` Dataset if necessary
new_dataset.push_to_hub("VikramSingh178/Products-10k-BLIP-captions")
