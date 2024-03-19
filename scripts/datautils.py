from datasets import load_dataset, Image
from config import Dataset_Name, DATA_DIR
from logger import rich_logger as l
import wandb
from config import Project_Name, entity
import pandas as pd
from tqdm import tqdm








class DatasetUtils:
    """
    Utility class for working with datasets.
    """
    def __init__(self, dataset_name:str,split:str=None):
        super().__init__()
        """
        Initializes a new instance of the DatasetUtils class.

        Args:
            dataset_name (str): The name of the dataset to use.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(self.dataset_name,cache_dir=DATA_DIR,split=split)
       
        self.dataset=self.dataset.remove_columns(['id'])
        l.info(f"Initialized dataset: {self.dataset_name}")
        l.info(self.dataset.features)
        

        
    
    

    


    
        
       
      
        
        
    
    
if __name__=="__main__":
    dataset = DatasetUtils(Dataset_Name,split="train")
    
    
   
       
    
