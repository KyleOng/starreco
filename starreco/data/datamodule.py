import pytorch_lightning as pl 
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

from starreco.data.dataset import *
from starreco.preprocessing import Preprocessor

class DataModule(pl.LightningDataModule):
    datasets = ["ml-1m", "epinions",  "book-crossing"]

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
        if self.dataset =="ml-1m":
             # Movielens datasets
            dataset = MovielensDataset()
        elif self.dataset == "epinions": 
             # Epinions dataset
            dataset = EpinionsDataset()
        elif self.dataset == "book-crossing": 
            # Book Crossing dataset
            dataset = BookCrossingDataset()

        ratings = dataset.prepare_data()

        #print(Preprocessor().df_to_sparse(ratings, dataset.user_column, dataset.item_column))
        #print(Preprocessor().transform(dataset.rated_items, return_dataframe = True))

        return ratings[[dataset.user_column, dataset.item_column]].values, \
        ratings[dataset.rating_column].values

    def setup(self, stage = None, random_state = 77):
        "Perform data operations"

        X, y = self.prepare_data()

        # General rule of thumb 60/20/20 train valid test split 
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size = 0.2, 
                                                            random_state = random_state) 

        X_valid, X_test, y_valid, y_test = train_test_split(X_test, 
                                                            y_test, 
                                                            test_size = 0.5, 
                                                            random_state = random_state)                                
        #print(X_train, X_test, y_train, y_test)
        

