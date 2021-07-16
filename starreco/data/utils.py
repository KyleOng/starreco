import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack
from deprecated import deprecated

# Done
def df_reindex(df, column):
    df = df.copy()
    df[column] = df[column].astype("category").cat.codes
    return df

# Done
def df_map(df, column):
    """
    Return a map dictionary which store reindexed column values.
    """
    return {v: i for i, v in enumerate(df[column].astype("category").cat.categories)}

# Done
def df_map_column(df, column, mapper, join = "left"):
    """
    Map `df[column]` with according to `mapper`.

    - mapper (dict): Mapping correspondence.
    - join (str): Type of table join to be performed between `df` and `df_map` 一 a dataframe created from `mapper`. Table join will be performed if `df[column].values` do not store the same shape and element as `mapper.values()`. Table join will also be disregard if set as None. Default: "left"
    """
    df = df.copy()
    df[column] = df[column].map(mapper)
    if join:
        # Perform table join if `df[column].values` do not store the same shape and element as `mapper.values()`
        if not np.array_equal(df[column].values, list(mapper.values())):
            # Create a dataframe from `map`
            df_map = pd.DataFrame(mapper.values(), columns = [column])
            # Table join between `df_map` and `df`
            df = df_map.merge(df, on = column, how = join)
    return df.sort_values(column)

# Done
def ratings_to_sparse_matrix(users, items, ratings, num_users, num_items):
    """
    Transform user-item ratings dataframe into sparse matrix.
    - users: List of users.
    - items: List of items.
    - ratings: List of ratings.
    - num_users: number of users.
    - num_items: number of items.

    :return: sparse rating matrix in csr format.
    """
    matrix = lil_matrix((num_users, num_items), dtype = "uint8")
    for user, item, rating in zip(users, items, ratings):
        matrix[user, item] = rating

    #breakpoint() # Evaluate matrix
    #[[e,i] for e, i in enumerate(matrix[0].toarray()[1]) if i > 0]

    return matrix.tocsr()

# Deprecated
@deprecated(version = "1.0.0", reason = "Sparse matrix transformed inside collate_fn.")
def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy sparse coo_matrix to pytorch sprase coo_tensor.
    - coo: scipy sparse coo_matrix.

    :return: pytorch sparse coo_trensor.
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)

# Done
def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform matrix to tensor.
    """
    return [torch.Tensor(vstack(tensor).toarray()) if isinstance(tensor[0], csr_matrix) 
            else torch.Tensor(tensor) for tensor in zip(*batch)]

# Done        
class MatrixDataset(Dataset):
    """
    Custom Dataset class for numpy array, csr matrix and coo matrix.
    """

    def __init__(self, *matrices):
        self.matrices = [matrix.tocsr() if isinstance(matrix, coo_matrix)  
                         else matrix for matrix in matrices]
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.matrices)

    def __len__(self):
        return self.matrices[0].shape[0]

