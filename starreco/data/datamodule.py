import pytorch_lightning as pl 
import torch
from scipy.sparse import hstack
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from starreco.data.dataset import *
from starreco.preprocessing import Preprocessor

class CFDataModule(pl.LightningDataModule):
    """
    Pytorch datamodule class for collaborative filtering.

    Parameters
    ----------
    :option (str): chosen dataset option. Default: "ml-1m"

    :batch_size (int): training/validation/testing batch size. Default: 256


    Attibutes
    ---------
    :dataset (Dataset): chosen dataset obtain from `option`

    :batch_size (int): training/validation/testing batch size.

    :_dataset_options (list): List of available supported dataset options.

    """

    _dataset_options = ["ml-1m", "epinions", "book-crossing"]
    preprocessor = Preprocessor()

    def __init__(self, option:str = "ml-1m", batch_size:int = 256):
        assert option in self._dataset_options, (f"'{option}' not include in prefixed dataset options. Choose from {self._dataset_options}.")
        
        # Download dataset
        if option == "ml-1m":
            self.dataset = MovielensDataset()
        elif option == "epinions": 
            self.dataset = EpinionsDataset()
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
        self.y = ratings[self.dataset.rating.rating_column].values

    def to_tensor(self):
        """
        Transform X and y from numpy/scipy array to pytorch tensor.
        """

        self.X = torch.tensor(self.X).type(torch.LongTensor)
        self.y = torch.tensor(self.y).type(torch.FloatTensor)

    def split(self, random_state = 77):
        """
        Perform train/validate/test split. 
        
        :random_state (int): A seed to sklearn's random number generator which to generate the splits. This ensures that the splits generated as reproducable. Default: 77
        """

        # General rule of thumb 60/20/20 train valid test split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, stratify = self.y, test_size = 0.2, random_state = random_state
        ) 
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            self.X_test, self.y_test, stratify = self.y_test, test_size = 0.5, random_state = random_state
        ) 

    def setup(self, stage = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        self.prepare_data()
        self.to_tensor()
        self.split()

    def train_dataloader(self):
        """
        Train dataloader.
        """

        train_ds = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_ds, batch_size = self.batch_size)
                          
    def val_dataloader(self):
        """
        Validate dataloader.
        """

        valid_ds = TensorDataset(self.X_valid, self.y_valid)
        return DataLoader(valid_ds, batch_size = self.batch_size)

    def test_dataloader(self):
        """
        Test dataloader.
        """

        test_ds = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_ds, batch_size = self.batch_size)
        
class CFSIDataModule(CFDataModule):
    """
    Pytorch datamodule class for collaborative filtering incorporated with side information.

    Parameters
    ----------
    :option (str): chosen dataset option. Default ml-1m

    :batch_size (int): training/validation/testing batch size.

    :user_item_indices (bool): If True, user and item indices (1st 2 columns of X) are included in X, else user and item indices are removed from X. 
    
    
    Attibutes
    ---------
    :dataset (Dataset): chosen dataset obtain from `option`

    :batch_size (int): training/validation/testing batch size.

    :user_item_indices (bool): If True, user and item indices are included in X, else user and item indices are removed from X.

    :_dataset_options (list): List of available supported dataset options.

    """

    def __init__(self, option = "ml-1m", batch_size = 256, user_item_indices = True):
        self.user_item_indices = user_item_indices
        super().__init__(option, batch_size)
    
    def prepare_data(self, stage = None):
        """
        Prepare X and y dataset. 
        """

        super().prepare_data()
        if not self.user_item_indices:
            self.X = None

        self.X = hstack([
            self.X, 
            self.preprocessor.transform(
                self.dataset.rating.user_select_related, 
                cat_columns = self.dataset.user.cat_columns,
                num_columns = self.dataset.user.num_columns, 
                set_columns = self.dataset.user.set_columns
            ), 
            self.preprocessor.transform(
                self.dataset.rating.item_select_related, 
                cat_columns = self.dataset.item.cat_columns,
                num_columns = self.dataset.item.num_columns, 
                set_columns = self.dataset.item.set_columns
            ), 
        ])

    def to_tensor(self):
        """
        Transform X and y from numpy/scipy array to pytorch dense/sparse tensor.
        """

        self.X_train = self.preprocessor.sparse_coo_to_tensor(self.X_train.tocoo())
        self.X_valid = self.preprocessor.sparse_coo_to_tensor(self.X_valid.tocoo())
        self.X_test = self.preprocessor.sparse_coo_to_tensor(self.X_test.tocoo())
        self.y_train = torch.tensor(self.y_train).type(torch.FloatTensor)
        self.y_valid = torch.tensor(self.y_valid).type(torch.FloatTensor)
        self.y_test = torch.tensor(self.y_test).type(torch.FloatTensor)

    def setup(self, stage = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        self.prepare_data()
        self.split()
        self.to_tensor()

class RCDataModule(CFDataModule):
    """
    Pytorch datamodule class for collaborative filtering reconstruction matrix.

    Parameters
    ----------
    :option (str): chosen dataset option. Default ml-1m

    :batch_size (int): training/validation/testing batch size.

    :transpose (bool): If True, transpose rating matrix, else pass.


    Attibutes
    ---------
    :dataset (Dataset): chosen dataset obtain from `option`

    :batch_size (int): training/validation/testing batch size.

    :transpose (bool): If True, transpose rating matrix, else pass.

    :_dataset_options (list): List of available supported dataset options.


    Notes
    -----
    Reconstruction matrix datamodule does not have y, since output = input (y = x).
    """

    def __init__(self, option = "ml-1m", batch_size = 256, transpose = False):
        self.transpose = transpose
        super().__init__(option, batch_size)

    def to_matrix(self):
        """
        Transform rating dataframe with M*N rows to rating matrix with M rows and N columns.
        """

        self.X_train = self.preprocessor.ratings_to_sparse(
            self.X_train.T[0], self.X_train.T[1], self.y_train, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_valid = self.preprocessor.ratings_to_sparse(
            self.X_valid.T[0], self.X_valid.T[1], self.y_valid, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )
        self.X_test = self.preprocessor.ratings_to_sparse(
            self.X_test.T[0], self.X_test.T[1], self.y_test, 
            self.dataset.rating.num_users, self.dataset.rating.num_items
        )

        if self.transpose:
            self.X_train = self.X_train.T
            self.X_valid = self.X_valid.T
            self.X_test = self.X_test.T

    def to_tensor(self):
        """
        Transform X from numpy/scipy array to pytorch dense/sparse tensor.
        """

        self.X_train = self.preprocessor.sparse_coo_to_tensor(self.X_train.tocoo())
        self.X_valid = self.preprocessor.sparse_coo_to_tensor(self.X_valid.tocoo())
        self.X_test = self.preprocessor.sparse_coo_to_tensor(self.X_test.tocoo())

    def setup(self, stage = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        self.prepare_data()
        self.split()
        self.to_matrix()
        self.to_tensor()

    def train_dataloader(self):
        """
        Train dataloader.
        """

        train_ds = TensorDataset(self.X_train, self.X_train)
        return DataLoader(train_ds, batch_size = self.batch_size)
                          
    def val_dataloader(self):
        """
        Validate dataloader.
        """

        valid_ds = TensorDataset(self.X_valid, self.X_valid)
        return DataLoader(valid_ds, batch_size = self.batch_size)

    def test_dataloader(self):
        """
        Test dataloader.
        """

        test_ds = TensorDataset(self.X_test, self.X_test)
        return DataLoader(test_ds, batch_size = self.batch_size)

class RCSIDataModule(RCDataModule):
    """
    Pytorch datamodule class for collaborative filtering reconstruction matrix incorporated with side information.

    Parameters
    ----------
    :option (str): chosen dataset option. Default ml-1m

    :batch_size (int): training/validation/testing batch size.

    :transpose (bool): If True, transpose rating matrix, else pass.


    Attibutes
    ---------
    :dataset (Dataset): chosen dataset obtain from `option`

    :batch_size (int): training/validation/testing batch size.

    :transpose (bool): If True, transpose rating matrix, else pass.

    :_dataset_options (list): List of available supported dataset options.


    Notes
    -----
    Reconstruction matrix datamodule does not have y, since output = input (y = x).

    """

    def add_side_information(self):
        """
        Incorporate side information to rating matrix.
        """

        if self.transpose:
            side_information = self.preprocessor.transform(
                self.dataset.item.map_column(self.dataset.rating.item_map, "left"), 
                cat_columns = self.dataset.item.cat_columns,
                num_columns = self.dataset.item.num_columns, 
                set_columns = self.dataset.item.set_columns
            )
        else:
            side_information = self.preprocessor.transform(
                self.dataset.user.map_column(self.dataset.rating.user_map, "left"), 
                cat_columns = self.dataset.user.cat_columns,
                num_columns = self.dataset.user.num_columns, 
                set_columns = self.dataset.user.set_columns
            )

        self.X_train = hstack([self.X_train, side_information])
        self.X_valid = hstack([self.X_valid, side_information])
        self.X_test = hstack([self.X_test, side_information])
    
    def setup(self, stage = None):
        """
        Data operation for each training/validating/testing.

        stage (str): Seperate setup logic for pytorch_lightining `trainer.fit` and `trainer.test`. Default: None
        """

        self.prepare_data()
        self.split()
        self.to_matrix()
        self.add_side_information()
        self.to_tensor()