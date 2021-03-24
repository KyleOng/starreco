import warnings

import pandas as pd
import numpy as np
import torch
from scipy.sparse import lil_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    """
    Original MultiLabelBinarizer fit_transform() only takes 2 positional arguments. 
    However, our custom pipeline assumes the MultiLabelBinarizer fit_transform() 
    is defined to take 3 positional arguments. Hence, adding an additional argument 
    y to fit_transform() fix the problem.
    """
    def fit_transform(self, X, y = None):
        """
        Fix original MultiLabelBinarizer fit_transform().

        :param X: X.

        :param y: y (target), set as None.

        :return: transformed X.      
        """
        y = None
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype = "object")
        return super().fit_transform(X.flatten())

class Preprocessor:
    def ratings_to_sparse(self, users, items, ratings, num_users, num_items):
        """
        Transform user-item ratings dataframe into sparse matrix.

        :param users: List of users.

        :param items: List of items.

        :param ratings: List of ratings.

        :param num_users: number of users.

        :param num_items: number of items.

        :return: sparse rating matrix.
        """
        matrix = lil_matrix((num_users, num_items), dtype = "uint8")
        for user, item, rating in zip(users, items, ratings):
            matrix[user, item] = rating

        #breakpoint() # Evaluate matrix
        #[[e,i] for e, i in enumerate(matrix[0].toarray()[1]) if i > 0]

        return matrix  

    def sparse_coo_to_tensor(self, coo):
        """
        Transform scipy sparse coo_matrix to pytorch sprase coo_tensor.

        :param coo: scipy sparse coo_matrix.

        :return: pytorch sparse coo_trensor.
        """
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def transform(self, df, cat_columns = [], num_columns = [], set_columns = [],
    return_dataframe = False):
        """
        Transform dataframe into algorithm compatible format.

        :param df: pandas dataframe.

        :param cat_columns: list of categorical columns exists in df.

        :param num_columns: list of numerical/continuous columns exists in df.

        :param set_columns: List of columns exists in df that contain sets.

        :param return_dataframe: If True return dataframe, else return sparse matrix. 

        :return: dataframe if return_dataframe set as True, else return sparse matrix.
        """
        # Return None if dataframe is empty
        if df is None:
            return None

        # Return None if cat/num/set_columns are empty
        if len(cat_columns) == 0 and len(num_columns) == 0 and len(set_columns) == 0:
            return None

        # Dynamic transformer/pipeline construction
        column_transformer = ColumnTransformer([])
        # Each column type has its own transformation pipeline
        # Categorical transformer: one hot encoder
        if len(cat_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ])
            column_transformer.transformers.append(
                ("categorical", pipe, cat_columns)
            )
        # Numerical transformer: min max scalar
        if len(num_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "mean")),
                ("onehot", MinMaxScaler())
            ])
            column_transformer.transformers.append(            
                ("numerical", pipe, num_columns)
            )
        # Set type transformers: multilabel binarizer
        if len(set_columns):
            # Each column has its own pipeline, because MultilabelBinarizer does not support multi column
            for set_column in set_columns: 
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
                    ("binarizer", CustomMultiLabelBinarizer(sparse_output = True))
                ])
                column_transformer.transformers.append(
                    (f"set_{set_column}", pipe, [set_column])
                )

        # Perform transformation/preprocessing
        df_transform = column_transformer.fit_transform(df)

        # Return transform data as dataframe if return_dataframe set as True
        if return_dataframe:
            try:
                columns_transform = []
                # Concatenate new transform columns
                if len(cat_columns):
                    columns_transform.append(column_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_columns))
                if len(num_columns):
                    columns_transform.append(numerical_columns)
                if len(set_columns):
                    for set_column in set_columns: 
                        columns_transform.append(
                            f"{set_column}_" + column_transformer.named_transformers_[f"set_{set_column}"].classes_
                        )
                return pd.DataFrame(df_transform.toarray(), columns = np.concatenate(columns_transform))
            except MemoryError as e:
                # Memory error. Return array instead
                warnings.warn(f"{str(e)}. Return values instead.")
                return df_transform
        else:
            return df_transform