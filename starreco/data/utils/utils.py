import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset


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


class MatrixDataset(TensorDataset):
    def __init__(self, *matrices):
        tensors = [torch.Tensor(matrix) for matrix in matrices]
        super().__init__(*tensors)
