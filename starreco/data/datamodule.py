import pytorch_lightning as pl 
from torchvision import transforms

from starreco.data.dataset import *

class DataModule(pl.LightningDataModule):
    datasets = [
        "ml-100k", 
        "ml-1m", 
        "ml-latest-small",
        "ml-latest",
        "epinions",
        "book-crossing"
    ]

    def __init__(self, dataset = "ml-1m"):
        """
        Constructor
        """
        # Validate whether predefined dataset exist
        if dataset in self.datasets: 
            self.dataset = dataset 
        else:
            raise Exception(f"'{dataset}' not include in prefixed dataset. Choose from {self.datasets}.")
        super().__init__()
    
    def prepare_data(self):
        """
        Prepare and download different dataset based on configuration from constructor
        """
        # Import dataset
        if "ml-" in self.dataset: 
             # Movielens datasets
            dataset = MovielensDataset(self.dataset.split("ml-")[-1])
        elif self.dataset =="epinions": 
             # Epinions dataset
            dataset = EpinionsDataset()
        elif self.dataset =="book-crossing": 
            # Book Crossing dataset
            dataset = BookCrossingDataset()
        df = dataset.import_data()

        return df

    def setup(self, stage = None):
        "Perform data operations"

        df = self.prepare_data()
        if stage == 'fit' or stage is None:
            pass
        if stage == 'test' or stage is None:
            pass