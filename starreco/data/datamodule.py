import pytorch_lightning as pl 
import torch
from scipy.sparse import hstack
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
            raise Exception(f"'{option}' ndot include in prefixed dataset options. Choose from {self.options}.")
        self.batch_size = batch_size
        super().__init__()
    
    def prepare_data(self):
        """
        Dataset selection
        """
        # Movielens datasets
        if self.option =="ml-1m":
            self.dataset = MovielensDataset()
        # Epinions dataset
        elif self.option == "epinions": 
            self.dataset = EpinionsDataset()
        # Book Crossing dataset
        elif self.option == "book-crossing": 
            self.dataset = BookCrossingDataset()

    def split(self, X, y, random_state = 77):
        # General rule of thumb 60/20/20 train valid test split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size = 0.2, random_state = random_state
        ) 
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            self.X_test, self.y_test, test_size = 0.5, random_state = random_state
        ) 

    def setup(self, stage = None):
        """
        Dataset setup: input-output (features-target) variables and train-validation-test split
        """
        self.prepare_data()

        X = self.dataset.ratings[[self.dataset.user_column, self.dataset.item_column]].values
        y = self.dataset.ratings[self.dataset.rating_column].values

        self.split(X, y)
        
class HybridDataModule(DataModule):
    def setup(self, stage = None):
        self.prepare_data()

        X = self.dataset.ratings[[self.dataset.user_column, self.dataset.item_column]].values
        y = self.dataset.ratings[self.dataset.rating_column].values

        preprocessor = Preprocessor()
        X = hstack([
            X,
            preprocessor.transform(self.dataset.rated_users),
            preprocessor.transform(self.dataset.rated_items)
        ])

        self.split(X, y)
        
class MatrixDataModule(DataModule):
    def __init__(self, option = "ml-1m", batch_size = None, transpose = False):
        self.transpose = transpose
        super().__init__(option, batch_size)

    def setup(self, stage = None):
        super().setup(stage)
        preprocessor = Preprocessor()

        num_users = self.dataset.num_users
        num_items = self.dataset.num_items

        self.X_train = preprocessor.ratings_to_sparse(
            self.X_train.T[0], self.X_train.T[1], self.y_train, num_users, num_items
        )
        self.X_valid = preprocessor.ratings_to_sparse(
            self.X_valid.T[0], self.X_valid.T[1], self.y_valid, num_users, num_items
        )
        self.X_test = preprocessor.ratings_to_sparse(
            self.X_test.T[0], self.X_test.T[1], self.y_test, num_users, num_items
        )

        if self.transpose:
            self.X_train = self.X_train.T
            self.X_valid = self.X_valid.T
            self.X_test = self.X_test.T

class HybridMatrixDataModule(MatrixDataModule):
    def setup(self, stage = None):
        super().setup(stage)
        preprocessor = Preprocessor()

        if self.transpose:
            side_info = self.dataset.items.dropna(subset = [self.dataset.item_column])
            side_info = side_info[self.dataset.rated_items.columns]
        else:
            side_info = self.dataset.users.dropna(subset = [self.dataset.user_column])
            side_info = side_info[self.dataset.rated_users.columns]

        side_info = preprocessor.transform(side_info)
        self.X_train = hstack([self.X_train, side_info])
        self.X_valid = hstack([self.X_valid, side_info])
        self.X_test = hstack([self.X_test, side_info])
