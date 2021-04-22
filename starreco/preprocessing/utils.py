import numpy as np
import torch
from scipy.sparse import lil_matrix, coo_matrix

def ratings_to_sparse(users, items, ratings, num_users, num_items):
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