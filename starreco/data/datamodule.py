import numpy as np
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

from starreco.preprocessing import Preprocessor
from .dataset import BookCrossingDataset, MovielensDataset
from .utils import MatrixDataset, sparse_batch_collate, ratings_to_sparse_matrix, df_map_column

# Done
class StarDataModule(pl.LightningDataModule):
    """
    Custom starrreco DataModule class.

    download (str): Dataset to be downloaded. Default "ml-1m".
    batch_size (int): Batch size.
    matrix_form (bool): Transform dataset to matrix form. Default: False.
    matrix_tranpose (bool): Transpose matrix data, `matrix_form` must be set to True. Default: False.
    add_ids (list): Add ids to matrix dataset, `matrix_form` must be set to True. Default: False.
    add_features (bool): Add user/item features to dataset. Default: False.
    user_features_ignore (list): User features to be ignored. Default: [].
    item_features_ignore (list): Item features to be ignored. Default: [].
    train_val_test_split (list): Train validation testing split ratio. Default: [60, 20, 20].
    num_workers (int): Number of CPU workers for Matrix Dataset. Default: 8.
    """

    _downloads = ["ml-1m", "book-crossing"]

    def __init__(self, 
                 download:str = "ml-1m",
                 batch_size:int = 256,
                 matrix_form:bool = False,
                 matrix_transpose:bool = False,
                 add_ids:bool = False,
                 add_features:bool = False,
                 user_features_ignore:list = [],
                 item_features_ignore:list = [],
                 train_val_test_split:list = [60, 20 ,20],
                 num_workers:int = 8):
        assert download in self._downloads, (f"`download` = '{download}' not include in prefixed dataset downloads. Choose from {self._downloads}.")
        assert not(not matrix_form and matrix_transpose), ("`matrix_form` and 'matrix_transpose` must be either False:False, True:False or True:True, cannot be False:True.")
        assert not(not matrix_form and add_ids), ("`matrix_form` and 'add_ids` must be either False:False, True:False or True:True, cannot be False:True.")
        assert sum(train_val_test_split) == 100, "The sum of `train_val_test_split` must be 100."

        self.batch_size = batch_size
        self.matrix_form = matrix_form
        self.matrix_transpose = matrix_transpose
        self.add_ids = add_ids
        self.add_features = add_features
        self.user_features_ignore = user_features_ignore
        self.item_features_ignore = item_features_ignore
        self.train_split, self.val_split, self.test_split = train_val_test_split
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

        # Include features to batch
        if self.add_features:
            # user preprocessed features data            
            user = df_map_column(self.dataset.user.df, self.dataset.user.column, self.dataset.rating.user_map, "left")
            user_cat_columns = list(set(self.dataset.user.cat_columns) - set(self.user_features_ignore))
            user_num_columns = list(set(self.dataset.user.num_columns) - set(self.user_features_ignore))
            user_set_columns = list(set(self.dataset.user.set_columns) - set(self.user_features_ignore))
            user_doc_columns = list(set(self.dataset.user.doc_columns) - set(self.user_features_ignore))
            self.user_preprocessor = Preprocessor(user, user_cat_columns, user_num_columns, user_set_columns, user_doc_columns)
            self.user = self.user_preprocessor.transform()

            # item preprocessed features data
            item = df_map_column(self.dataset.item.df, self.dataset.item.column, self.dataset.rating.item_map, "left")
            item_cat_columns = list(set(self.dataset.item.cat_columns) - set(self.item_features_ignore))
            item_num_columns = list(set(self.dataset.item.num_columns) - set(self.item_features_ignore))
            item_set_columns = list(set(self.dataset.item.set_columns) - set(self.item_features_ignore))
            item_doc_columns = list(set(self.dataset.item.doc_columns) - set(self.item_features_ignore))
            self.item_preprocessor = Preprocessor(item, item_cat_columns, item_num_columns, item_set_columns, item_doc_columns)
            self.item = self.item_preprocessor.transform()

            # map user and item to ratings
            self.user_X = self.user[self.X[:, 0]]
            self.item_X = self.item[self.X[:, 1]]

    def split(self, random_state:int = 77):
        """
        Perform train/validate/test split. 

        Note: General rule of thumb 60/20/20 train valid test split 
        """
        # Ttrain validation split ratio
        train_val_split = (self.val_split + self.test_split) / (self.train_split + self.val_split + self.test_split)
        # Validation test split ratio
        val_test_split = self.val_split/(self.val_split + self.test_split)
        
        if not self.matrix_form and self.add_features:
            self.X_train, self.X_val, self.user_X_train, self.user_X_val, self.item_X_train, self.item_X_val, self.y_train, self.y_val \
            = train_test_split(self.X, self.user_X, self.item_X, self.y, stratify = self.y, test_size = train_val_split, random_state = random_state)
            if self.test_split:
                self.X_val, self.X_test, self.user_X_val, self.user_X_test, self.item_X_val, self.item_X_test, self.y_val, self.y_test \
                = train_test_split(self.X_val, self.user_X_val, self.item_X_val, self.y_val, stratify = self.y_val, test_size = val_test_split, random_state = random_state) 
        else:
            self.X_train, self.X_val, self.y_train, self.y_val \
            = train_test_split(self.X, self.y, stratify = self.y, test_size = train_val_split, random_state = random_state) 
            if self.test_split:
                self.X_val, self.X_test, self.y_val, self.y_test \
                = train_test_split(self.X_val, self.y_val, stratify = self.y_val, test_size = val_test_split, random_state = random_state)                

    def to_matrix(self):
        """
        Transform rating dataframe with M*N rows to rating matrix with M rows and N columns.
        """
        self.X_train = ratings_to_sparse_matrix(self.X_train[:, 0], self.X_train[:, 1], self.y_train, self.dataset.rating.num_users,  self.dataset.rating.num_items)
        self.X_val = ratings_to_sparse_matrix(self.X_val[:, 0], self.X_val[:, 1], self.y_val, self.dataset.rating.num_users, self.dataset.rating.num_items)
        if self.test_split:
            self.X_test = ratings_to_sparse_matrix(self.X_test[:, 0], self.X_test[:, 1], self.y_test, self.dataset.rating.num_users, self.dataset.rating.num_items)

        # Transpose X
        if self.matrix_transpose:
            self.X_train = self.X_train.T
            self.X_val = self.X_val.T
            if self.test_split:
                self.X_test = self.X_test.T

        # Since reconstruction input as y, set y as x
        self.y_train = self.X_train
        self.y_val = self.X_val
        if self.test_split:
            self.y_test = self.X_test

    def setup(self, stage = None):
        """
        Data operation for each training/validating/testing.
        """
        self.prepare_data()
        self.split()
        if self.matrix_form: 
            self.to_matrix()

    def train_dataloader(self):
        """
        Train dataloader.
        """
        train_data = [self.X_train, self.y_train]

        if self.add_features:
            if self.matrix_form:
                if self.matrix_transpose:
                    train_data.insert(-1, self.item)
                else:
                    train_data.insert(-1, self.user)
            else:
                train_data.insert(-1, self.user_X_train)
                train_data.insert(-1, self.item_X_train)

        if self.add_ids and self.matrix_form:
            train_data.insert(-1, np.arange(self.X_train.shape[0]))

        train_ds = MatrixDataset(*train_data)
        train_dl = DataLoader(train_ds, 
                              batch_size = self.batch_size, 
                              collate_fn = sparse_batch_collate, 
                              num_workers = self.num_workers)

        return train_dl
                          
    def val_dataloader(self):
        """
        Validation dataloader.
        """
        val_data = [self.X_val, self.y_val]

        if self.add_features:
            if self.matrix_form:
                if self.matrix_transpose:
                    val_data.insert(-1, self.item)
                else:
                    val_data.insert(-1, self.user)
            else:
                val_data.insert(-1, self.user_X_val)
                val_data.insert(-1, self.item_X_val)

        if self.add_ids and self.matrix_form:
            val_data.insert(-1, np.arange(self.X_val.shape[0]))

        val_ds = MatrixDataset(*val_data)
        val_dl = DataLoader(val_ds, 
                            batch_size = self.batch_size, 
                            collate_fn = sparse_batch_collate, 
                            num_workers = self.num_workers)
        return val_dl

    def test_dataloader(self):
        """
        Testing dataloader.
        """
        if self.test_split:
            test_data = [self.X_test, self.y_test]

            if self.add_features:
                if self.matrix_form:
                    if self.matrix_transpose:
                        test_data.insert(-1, self.item)
                    else:
                        test_data.insert(-1, self.user)
                else:
                    test_data.insert(-1, self.user_X_test)
                    test_data.insert(-1, self.item_X_test)

            if self.add_ids and self.matrix_form:
                test_data.insert(-1, np.arange(self.X_test.shape[0]))

            test_ds = MatrixDataset(*test_data)
            test_dl = DataLoader(test_ds, 
                                batch_size = self.batch_size, 
                                collate_fn = sparse_batch_collate, 
                                num_workers = self.num_workers)

            return test_dl
        else:
            return None