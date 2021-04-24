from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack, issparse


def df_map_column(df:pd.DataFrame, column:str, arg:dict, join:str = "left"):
    """
    Map `df[column]` with according to `arg`.

    :param arg (dict): Mapping correspondence.

    :join (str): Type of table join to be performed between `df` and `df_map` 一 a dataframe created from `arg`. Table join will be performed if `df[column].values` do not store the same shape and element as `arg.values()`. Table join will also be disregard if set as None. Default: "left"
    """
    df = df.copy()
    df[column] = df[column].map(arg)
    if join:
        # Perform table join if `df[column].values` do not store the same shape and element as `arg.values()`
        if not np.array_equal(df[column].values, list(arg.values())):
            # Create a dataframe from `map`
            df_map = pd.DataFrame(arg.values(), columns = [column])
            # Table join between `df_map` and `df`
            df = df_map.merge(df, on = column, how = join)
    return df.sort_values(column)


def ratings_to_sparse_matrix(users, items, ratings, num_users, num_items):
    """
    Transform user-item ratings dataframe into sparse matrix.

    :param users: List of users.

    :param items: List of items.

    :param ratings: List of ratings.

    :param num_users: number of users.

    :param num_items: number of items.

    :return: sparse rating matrix in csr format.
    """
    matrix = lil_matrix((num_users, num_items), dtype = "uint8")
    for user, item, rating in zip(users, items, ratings):
        matrix[user, item] = rating

    #breakpoint() # Evaluate matrix
    #[[e,i] for e, i in enumerate(matrix[0].toarray()[1]) if i > 0]

    return matrix.tocsr()


def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy sparse coo_matrix to pytorch sprase coo_tensor.

    :param coo: scipy sparse coo_matrix.

    :return: pytorch sparse coo_trensor.
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    return [sparse_coo_to_tensor(vstack(data).tocoo()) for data in zip(*batch)]

            
class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, *sparses:Union[coo_matrix, csr_matrix]):
        self.sparses = [sparse.tocsr() if isinstance(sparse, coo_matrix)
                        else sparse 
                        for sparse in sparses]
        
    def __getitem__(self, index:int):
        return [sparse[index] for sparse in self.sparses]

    def __len__(self):
        return self.sparses[0].shape[0]


class MatrixDataset(TensorDataset):
    """
    Custom Dataset class for matrix type data, which later transform to `torch.Tensor` during initialization.
    """

    def __init__(self, *matrices:np.array):
        tensors = [torch.Tensor(matrix) for matrix in matrices]
        super().__init__(*tensors)
