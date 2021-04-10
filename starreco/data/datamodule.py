import pytorch_lightning as pl 
import torch
from scipy.sparse import hstack
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from starreco.preprocessing import Preprocessor, ratings_to_sparse
from .dataset import MovielensDataset, BookCrossingDataset
from .sparse import SparseDataset, sparse_batch_collate

class BaseDataModule(pl.LightningDataModule):
    """
    Base data module class

    Warning: This method should not be used directly.
    """

    _dataset_options = ["ml-1m", "book-crossing"]

    def __init__(self, option:str = "ml-1m", batch_size:int = 256):

        assert option in self._dataset_options, (f"'{option}' not include in prefixed dataset options. Choose from {self._dataset_options}.")
        
        # Download dataset
        if option == "ml-1m":
            self.dataset = MovielensDataset()
        elif option == "book-crossing": 
            self.dataset = BookCrossingDataset()

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


    def split(self, random_state:int = 77):
        """
        Perform train/validate/test split. 
        
        :param random_state (int): A seed to sklearn's random number generator which to generate the splits. This ensures that the splits generated as reproducable. Default: 77
        """

        # General rule of thumb 60/20/20 train valid test split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, stratify = self.y, test_size = 0.2, random_state = random_state
        ) 
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            self.X_test, self.y_test, stratify = self.y_test, test_size = 0.5, random_state = random_state
        ) 

    def setup(self, stage:str = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        self.prepare_data()
        self.split()

    def train_dataloader(self):
        """
        Train dataloader.
        """

        train_ds = SparseDataset(self.X_train, self.y_train)
        return DataLoader(train_ds, batch_size = self.batch_size)
                          
    def val_dataloader(self):
        """
        Validate dataloader.
        """

        valid_ds = SparseDataset(self.X_valid, self.y_valid)
        return DataLoader(valid_ds, batch_size = self.batch_size)

    def test_dataloader(self):
        """
        Test dataloader.
        """

        test_ds = SparseDataset(self.X_test, self.y_test)
        return DataLoader(test_ds, batch_size = self.batch_size)

class SparseDataModule(BaseDataModule):
    def train_dataloader(self):
        """
        Train dataloader for sparse dataset.
        """

        train_ds = SparseDataset(self.X_train, self.y_train)
        return DataLoader(train_ds, 
                          batch_size = self.batch_size, 
                          collate_fn = sparse_batch_collate)
                          
    def val_dataloader(self):
        """
        Validate dataloader for sparse dataset.
        """

        valid_ds = SparseDataset(self.X_valid, self.y_valid)
        return DataLoader(valid_ds, 
                          batch_size = self.batch_size, 
                          collate_fn = sparse_batch_collate)

    def test_dataloader(self):
        """
        Test dataloader for sparse dataset.
        """

        test_ds = SparseDataset(self.X_test, self.y_test)
        return DataLoader(test_ds, 
                          batch_size = self.batch_size, 
                          collate_fn = sparse_batch_collate)

class CFDataModule(BaseDataModule):
    """
    Pytorch datamodule class for collaborative filtering.

    :param option (str): chosen dataset option. Default: "ml-1m"

    :param batch_size (int): training/validation/testing batch size. Default: 256
    """
    
    def to_tensor(self):
        """
        Transform X and y from numpy/scipy array to pytorch tensor.
        """

        self.X_train = torch.tensor(self.X_train).type(torch.LongTensor)
        self.X_valid = torch.tensor(self.X_valid).type(torch.LongTensor)
        self.X_test = torch.tensor(self.X_test).type(torch.LongTensor)
        self.y_train = torch.tensor(self.y_train).type(torch.FloatTensor)
        self.y_valid = torch.tensor(self.y_valid).type(torch.FloatTensor)
        self.y_test = torch.tensor(self.y_test).type(torch.FloatTensor)
        
    def setup(self, stage:str = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        super().setup(stage)
        self.to_tensor()
    

class CFSIDataModule(SparseDataModule):
    """
    Pytorch datamodule class for collaborative filtering incorporated with features (side information).

    :param option (str): chosen dataset option. Default: "ml-1m"

    :param batch_size (int): training/validation/testing batch size. Default: 256

    :param include_user_item (bool): If True, user and item fields (1st 2 columns of X) are included in X, else user and item fields are removed from X. 
    """

    def __init__(self, option:str = "ml-1m", batch_size:int = 256, include_user_item:bool = True):

        self.include_user_item = include_user_item
        super().__init__(option, batch_size)
    
    def prepare_data(self, stage:str = None):
        """
        Prepare X and y dataset. 
        """

        super().prepare_data()

        user_preprocessor = Preprocessor(
            self.dataset.rating.user_select_related,
            cat_columns = self.dataset.user.cat_columns,
            num_columns = self.dataset.user.num_columns, 
            set_columns = self.dataset.user.set_columns
        )
        user_features = user_preprocessor.transform()

        item_preprocessor = Preprocessor(
            self.dataset.rating.item_select_related,
            cat_columns = self.dataset.item.cat_columns,
            num_columns = self.dataset.item.num_columns, 
            set_columns = self.dataset.item.set_columns
        )
        item_features = item_preprocessor.transform()

        self.feature_dims = [user_features.shape[1], item_features.shape[1]]
        
        if self.include_user_item:
            self.X = hstack([self.X, user_features, item_features])
        else:
            self.X = hstack([user_features, item_features])
        
class RCDataModule(SparseDataModule):
    """
    Pytorch datamodule class for collaborative filtering reconstruction matrix.


    :param option (str): chosen dataset option. Default: "ml-1m"

    :param batch_size (int): training/validation/testing batch size. Default: 256

    :param transpose (bool): If True, transpose rating matrix, else pass.

    Note: Reconstruction matrix datamodule does not have y, since output = input (y = x).
    """

    def __init__(self, option:str = "ml-1m", batch_size:int = 256, transpose:bool = False):

        self.transpose = transpose
        super().__init__(option, batch_size)

    def to_matrix(self):
        """
        Transform rating dataframe with M*N rows to rating matrix with M rows and N columns.
        """

        self.X_train = ratings_to_sparse(
            self.X_train.T[0], self.X_train.T[1], self.y_train, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_valid = ratings_to_sparse(
            self.X_valid.T[0], self.X_valid.T[1], self.y_valid, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_test = ratings_to_sparse(
            self.X_test.T[0], self.X_test.T[1], self.y_test, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )

        if self.transpose:
            self.X_train = self.X_train.T
            self.X_valid = self.X_valid.T
            self.X_test = self.X_test.T
        
    def setup(self, stage:str = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        super().setup(stage)
        self.to_matrix()

        # Since reconstruction input = output, set y = x
        self.y_train = self.X_train
        self.y_valid = self.X_valid
        self.y_test = self.X_test

class RCSIDataModule(RCDataModule):
    """
    Pytorch datamodule class for collaborative filtering reconstruction matrix incorporated with features (side information).

    :param option (str): chosen dataset option. Default: "ml-1m"

    :param batch_size (int): training/validation/testing batch size. Default: 256

    :param transpose (bool): If True, transpose rating matrix, else pass.

    Notes: Reconstruction matrix datamodule does not have y, since output = input (y = x).

    """

    def add_features(self):
        """
        Incorporate user or item features (side information) to rating matrix.
        """

        if self.transpose:
            preprocessor = Preprocessor(
                df = self.dataset.item.map_column(self.dataset.rating.item_map, "left"), 
                cat_columns = self.dataset.item.cat_columns,
                num_columns = self.dataset.item.num_columns, 
                set_columns = self.dataset.item.set_columns
            )
        else:
            preprocessor = Preprocessor(
                df = self.dataset.user.map_column(self.dataset.rating.user_map, "left"), 
                cat_columns = self.dataset.user.cat_columns,
                num_columns = self.dataset.user.num_columns, 
                set_columns = self.dataset.user.set_columns
            )
            
        features = preprocessor.transform()
        self.feature_dim = features.shape[1]

        self.X_train = hstack([self.X_train, features])
        self.X_valid = hstack([self.X_valid, features])
        self.X_test = hstack([self.X_test, features])

        self.y_train = hstack([self.y_train, features])
        self.y_valid = hstack([self.y_valid, features])
        self.y_test = hstack([self.y_test, features])
    
    def setup(self, stage:str = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        super().setup(stage)
        self.add_features()