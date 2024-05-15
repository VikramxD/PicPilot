from datasets import load_dataset
from config import (PRODUCTS_10k_DATASET,CAPTIONING_MODEL_NAME)
from transformers import (BlipProcessor, BlipForConditionalGeneration)
from tqdm import tqdm
import torch




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ImageCaptioner:
    
    def __init__(self, dataset:str,processor:str,model:str):
        self.dataset = load_dataset(dataset)
        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model).to(device)
    
    
    def process_dataset(self):
        self.dataset = self.dataset.rename_column(original_column_name='pixel_values',new_column_name='image')
        self.dataset = self.dataset.remove_columns(column_names=['label'])
        return self.dataset
    
    
    def generate_captions(self):
        self.dataset = self.process_dataset()
        self.dataset['image']=[image.convert("RGB") for image in self.dataset["image"]]
        print(self.dataset['image'][0])
        for image in tqdm(self.dataset['image']):
            inputs = self.processor(image, return_tensors="pt").to(device)
            out = self.model(**inputs)
            

            
       
       
       
       
       
       
ic = ImageCaptioner(dataset=PRODUCTS_10k_DATASET,processor=CAPTIONING_MODEL_NAME,model=CAPTIONING_MODEL_NAME)


        
        







