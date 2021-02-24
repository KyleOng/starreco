import pytorch_lightning as pl 
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

from starreco.data.dataset import *
from starreco.preprocessing import Preprocessor

class DataModule(pl.LightningDataModule):
    options = ["ml-1m", "epinions",  "book-crossing"]

    def __init__(self, option = "ml-1m", batch_size = None):
        """
        Constructor
        """
        # Validate whether predefined dataset exist
        if option in self.options: 
            self.option = option
        else:
            raise Exception(f"'{option}' not include in prefixed dataset options. Choose from {self.options}.")
        self.batch_size = batch_size
        super().__init__()
    
    def prepare_data(self):
        """
        Prepare and download different dataset based on configuration from constructor
        """
        # Import dataset
        # Movielens datasets
        if self.option =="ml-1m":
            dataset = MovielensDataset()
        # Epinions dataset
        elif self.option == "epinions": 
            dataset = EpinionsDataset()
        # Book Crossing dataset
        elif self.option == "book-crossing": 
            dataset = BookCrossingDataset()
        self.dataset = dataset
        ratings = dataset.prepare_data()

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

        #print(Preprocessor().ratings_to_sparse(X_train.T[0], X_train.T[1], y_train, self.dataset.num_users, self.dataset.num_items))

 class MatrixDataModule(DataModule):
     pass       


