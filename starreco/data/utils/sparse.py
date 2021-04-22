from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack


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


def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.Tensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.Tensor(targets_batch)
    return data_batch, targets_batch


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


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, 
                 data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 targets:Union[np.ndarray, coo_matrix, csr_matrix]
                 ):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.data.shape[0]