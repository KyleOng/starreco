from typing import Union

import torch
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, coo_matrix, issparse

from starreco.preprocessing import Preprocessor
from .dataset import BookCrossingDataset, MovielensDataset
from .utils import MatrixDataset, SparseDataset, sparse_coo_to_tensor, sparse_batch_collate, ratings_to_sparse_matrix, df_map_column


class StarDataModule(pl.LightningDataModule):
    """
    starreco custom DataModule class.
    """

    _downloads = ["ml-1m", "book-crossing"]

    def __init__(self, 
                 download:str = "ml-1m",
                 batch_size:int = 256,
                 features_join:bool = False,
                 matrix_transform:bool = False,
                 matrix_transpose:bool = False,
                 num_workers:int = 0):
        assert download in self._downloads, \
        (f"`download` = '{download}' not include in prefixed dataset downloads. Choose from {self._downloads}.")
        
        assert not(not matrix_transform and matrix_transpose), \
        ("`matrix_transform` and 'matrix_transpose` must be either False:False, True:False or True:True, cannot be False:True.")

        self.batch_size = batch_size
        self.features_join = features_join
        self.matrix_transform = matrix_transform
        self.matrix_transpose = matrix_transpose
        self.num_workers = num_workers
        
        # Download dataset
        if download == "ml-1m": 
            self.dataset = MovielensDataset()
        elif download == "book-crossing": 
            self.dataset = BookCrossingDataset()

        super().__init__()
    
    def prepare_data(self):
        """
        Prepare X and y dataset, along with user and item side information.
        """
        ratings = self.dataset.rating.reindex

        self.X = ratings[[self.dataset.user.column, self.dataset.item.column]].values
        self.y = ratings[self.dataset.rating.column].values

        # user and item preprocessed features data
        self.user = Preprocessor(
            df = df_map_column(
                self.dataset.user.df, self.dataset.user.column, 
                self.dataset.rating.user_map, 
                "left"
            ), 
            cat_columns = self.dataset.user.cat_columns,
            num_columns = self.dataset.user.num_columns, 
            set_columns = self.dataset.user.set_columns
        ).transform()

        self.item = Preprocessor(
            df = df_map_column(
                self.dataset.item.df, 
                self.dataset.item.column, 
                self.dataset.rating.item_map, 
                "left"
            ), 
            cat_columns = self.dataset.item.cat_columns,
            num_columns = self.dataset.item.num_columns, 
            set_columns = self.dataset.item.set_columns
        ).transform()

        # user_X and item_X, which are features mapped to X, only accessable for non-matrix X.
        if not self.matrix_transform and self.features_join:
            self.user_X = Preprocessor(
                self.dataset.rating.user_select_related,
                cat_columns = self.dataset.user.cat_columns,
                num_columns = self.dataset.user.num_columns, 
                set_columns = self.dataset.user.set_columns
            ).transform()

            self.item_X = Preprocessor(
                self.dataset.rating.item_select_related,
                cat_columns = self.dataset.item.cat_columns,
                num_columns = self.dataset.item.num_columns, 
                set_columns = self.dataset.item.set_columns
            ).transform()

    def split(self, random_state:int = 77):
        """
        Perform train/validate/test split. 
        
        :param random_state (int): A seed to sklearn's random number generator which to generate the splits. This ensures that the splits generated as reproducable. Default: 77

        Note: General rule of thumb 60/20/20 train valid test split 
        """
        if not self.matrix_transform and self.features_join:
            self.X_train, self.X_test, self.user_X_train, self.user_X_test, \
            self.item_X_train, self.item_X_test, self.y_train, self.y_test = \
            train_test_split(
                self.X, 
                self.user_X, 
                self.item_X, 
                self.y, 
                stratify = self.y, 
                test_size = 0.2, 
                random_state = random_state
            ) 
            self.X_val, self.X_test, self.user_X_val, self.user_X_test, \
            self.item_X_val, self.item_X_test, self.y_val, self.y_test = \
            train_test_split(
                self.X_test, 
                self.user_X_test, 
                self.item_X_test, 
                self.y_test, 
                stratify = self.y_test, 
                test_size = 0.5, 
                random_state = random_state
            ) 
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, 
                self.y, 
                stratify = self.y, 
                test_size = 0.2, 
                random_state = random_state
            ) 
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                self.X_test, 
                self.y_test, 
                stratify = self.y_test, 
                test_size = 0.5, 
                random_state = random_state
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
        if self.matrix_transform: 
            self.to_matrix()

    def train_dataloader(self):
        """
        Train dataloader.
        """
        train_data = [self.X_train, self.y_train]

        if self.features_join:
            if self.matrix_transform:
                if not self.matrix_transpose:
                    train_data.insert(1, self.user)
                else:
                    train_data.insert(1, self.item)
            else:
                train_data.insert(1, self.user_X_train)
                train_data.insert(2, self.item_X_train)

        train_dls = []
        for train_datum in train_data:
            if issparse(train_datum):
                train_ds = SparseDataset(train_datum)
                train_dl = DataLoader(train_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
            else:
                train_ds = MatrixDataset(train_datum)
                train_dl = DataLoader(train_ds, batch_size = self.batch_size, num_workers = self.num_workers)
            train_dls.append(train_dl)

        return zip(*train_dls)
                          
    def val_dataloader(self):
        """
        Validation dataloader.
        """
        val_data = [self.X_val, self.y_val]

        if self.features_join:
            if self.matrix_transform:
                if not self.matrix_transpose:
                    val_data.insert(1, self.user)
                else:
                    val_data.insert(1, self.item)
            else:
                val_data.insert(1, self.user_X_val)
                val_data.insert(2, self.item_X_val)

        val_dls = []
        for val_datum in val_data:
            if issparse(val_datum):
                val_ds = SparseDataset(val_datum)
                val_dl = DataLoader(val_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
            else:
                val_ds = MatrixDataset(val_datum)
                val_dl = DataLoader(val_ds, batch_size = self.batch_size, num_workers = self.num_workers)
            val_dls.append(val_dl)

        return zip(*val_dls)

    def test_dataloader(self):
        """
        Testing dataloader.
        """
        test_data = [self.X_test, self.y_test]

        if self.features_join:
            if self.matrix_transform:
                if not self.matrix_transpose:
                    test_data.insert(1, self.user)
                else:
                    test_data.insert(1, self.item)
            else:
                test_data.insert(1, self.user_X_test)
                test_data.insert(2, self.item_X_test)

        test_dls = []
        for test_datum in test_data:
            if issparse(test_datum):
                test_ds = SparseDataset(test_datum)
                test_dl = DataLoader(test_ds, batch_size = self.batch_size, collate_fn = sparse_batch_collate)
            else:
                test_ds = MatrixDataset(test_datum)
                test_dl = DataLoader(test_ds, batch_size = self.batch_size, num_workers = self.num_workers)
            test_dls.append(test_dl)

        return zip(*test_dls)