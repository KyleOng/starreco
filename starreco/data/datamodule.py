import pytorch_lightning as pl 
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, coo_matrix

from starreco.preprocessing import Preprocessor
from .dataset import BookCrossingDataset, MovielensDataset
from .utils import MatrixDataset, SparseDataset, sparse_coo_to_tensor, sparse_batch_collate, ratings_to_sparse_matrix, df_map_column

class StarDataModule(pl.LightningDataModule):
    """
    starreco custom DataModule class.
    """

    _dataset_options = ["ml-1m", "book-crossing"]

    def __init__(self, 
                 option:str = "ml-1m", 
                 matrix_form:bool = False,
                 matrix_transpose:bool = False,
                 add_features:bool = False,
                 batch_size:int = 256):

        assert option in self._dataset_options, (f"'{option}' not include in prefixed dataset options. Choose from {self._dataset_options}.")
        
        # Download dataset
        if option == "ml-1m":
            self.dataset = MovielensDataset()
        elif option == "book-crossing": 
            self.dataset = BookCrossingDataset()

        self.matrix_form = matrix_form
        self.matrix_transpose = matrix_transpose
        self.add_features = add_features
        self.batch_size = batch_size
        super().__init__()
    
    def prepare_data(self):
        """
        Prepare X and y dataset. 
        """
        ratings = self.dataset.rating.reindex

        self.X = ratings[[self.dataset.user.column, self.dataset.item.column]].values
        self.y = ratings[self.dataset.rating.column].values

        self.field_dims = [self.dataset.rating.num_users, self.dataset.rating.num_items]

        if self.add_features:
            user_preprocessor = Preprocessor(
                df = df_map_column(
                    self.dataset.user.df, self.dataset.user.column, self.dataset.rating.user_map, "left"
                ), 
                cat_columns = self.dataset.user.cat_columns,
                num_columns = self.dataset.user.num_columns, 
                set_columns = self.dataset.user.set_columns
            )
            item_preprocessor = Preprocessor(
                df = df_map_column(
                    self.dataset.item.df, self.dataset.item.column, self.dataset.rating.item_map, "left"
                ), 
                cat_columns = self.dataset.item.cat_columns,
                num_columns = self.dataset.item.num_columns, 
                set_columns = self.dataset.item.set_columns
            )
            self.user_X = user_preprocessor.transform()
            self.item_X = item_preprocessor.transform()

    def split(self, random_state:int = 77):
        """
        Perform train/validate/test split. 
        
        :param random_state (int): A seed to sklearn's random number generator which to generate the splits. This ensures that the splits generated as reproducable. Default: 77
        """
        # General rule of thumb 60/20/20 train valid test split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, stratify = self.y, test_size = 0.2, random_state = random_state
        ) 
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test, self.y_test, stratify = self.y_test, test_size = 0.5, random_state = random_state
        ) 

    def to_matrix(self):
        """
        Transform rating dataframe with M*N rows to rating matrix with M rows and N columns.
        """
        self.X_train = ratings_to_sparse_matrix(
            self.X_train.T[0], self.X_train.T[1], self.y_train, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_val = ratings_to_sparse_matrix(
            self.X_val.T[0], self.X_val.T[1], self.y_val, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_test = ratings_to_sparse_matrix(
            self.X_test.T[0], self.X_test.T[1], self.y_test, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        
        # Transpose X
        if self.matrix_transpose:
            self.X_train = self.X_train.T
            self.X_val = self.X_val.T
            self.X_test = self.X_test.T

        # Since reconstruction input = output, set y = x
        self.y_train = self.X_train
        self.y_val = self.X_val
        self.y_test = self.X_test

    def setup(self, stage:str = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """
        self.prepare_data()
        self.split()
        if self.matrix_form:
            self.to_matrix()      

    def train_dataloader(self):
        """
        Train dataloader.
        """
        if type(self.X_train) in [csr_matrix, coo_matrix] or type(self.y_train) in [csr_matrix, coo_matrix]:
            train_ds = SparseDataset(self.X_train, self.y_train)
            train_dl = DataLoader(train_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
        else:
            train_ds = MatrixDataset(self.X_train, self.y_train)
            train_dl = DataLoader(train_ds, batch_size = self.batch_size)

        return train_dl
                          
    def val_dataloader(self):
        """
        Validate dataloader.
        """
        if type(self.X_val) in [csr_matrix, coo_matrix] or type(self.y_val) in [csr_matrix, coo_matrix]:
            val_ds = SparseDataset(self.X_val, self.y_val)
            val_dl = DataLoader(val_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
        else:
            val_ds = MatrixDataset(self.X_val, self.y_val)
            val_dl = DataLoader(val_ds, batch_size = self.batch_size)

        return val_dl

    def test_dataloader(self):
        """
        Test dataloader.
        """
        if type(self.X_test) in [csr_matrix, coo_matrix] or type(self.y_test) in [csr_matrix, coo_matrix]:
            test_ds = SparseDataset(self.X_test, self.y_test)
            test_dl = DataLoader(test_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
        else:
            test_ds = MatrixDataset(self.X_test, self.y_test)
            test_dl = DataLoader(test_ds, batch_size = self.batch_size)

        return test_dl
