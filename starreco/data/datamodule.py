import pytorch_lightning as pl 

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
        """Constructor"""

        # Validate whether predefined dataset exist
        if dataset in self.datasets: 
            self.dataset = dataset 
        else:
            raise Exception(f"'{dataset}' not include in prefixed dataset. Choose from {self.datasets}.")
        super().__init__()
    
    def prepare_data(self):
        "Prepare data from different dataset"

        # Movielens datasets
        if "ml-" in self.dataset:
            df = MovielensDataset(self.dataset.split("ml-")[-1]).import_data()

        # Epinions dataset
        elif self.dataset =="epinions":
            df = EpinionsDataset().import_data()

        # Book Crossing dataset
        elif self.dataset =="book-crossing":
            df = BookCrossingDataset().import_data()

        return df