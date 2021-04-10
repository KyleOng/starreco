import warnings

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    """
    Custom MultiLabelBinarizer.

    Notes: Original MultiLabelBinarizer fit_transform() only takes 2 positional arguments. However, our custom pipeline assumes the MultiLabelBinarizer fit_transform() is defined to take 3 positional arguments. Hence, adding an additional argument y to fit_transform() fix the problem.
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
    """
    Preprocessing class.

    :param df: pandas dataframe.

    :param cat_columns: list of categorical columns exists in df.

    :param num_columns: list of numerical/continuous columns exists in df.

    :param set_columns: List of columns exists in df that contain sets.
    """
    
    def __init__(self, df:pd.DataFrame, 
                 cat_columns:list = [], 
                 num_columns:list = [], 
                 set_columns:list = []):
        self.df = df
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.set_columns = set_columns

    def transform(self, return_dataframe = False):
        """
        Transform dataframe into algorithm compatible format.

        :return: dataframe if return_dataframe set as True, else return sparse matrix. Default: False
        """

        # Return None if dataframe is empty
        if self.df is None:
            return None

        # Return None if cat/num/self.set_columns are empty
        if len(self.cat_columns) == 0 and \
        len(self.num_columns) == 0 and \
        len(self.set_columns) == 0:
            return None

        # Dynamic transformer/pipeline construction
        self.column_transformer = ColumnTransformer([])
        # Each column type has its own transformation pipeline
        # Categorical transformer: one hot encoder
        if len(self.cat_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ])
            self.column_transformer.transformers.append(
                ("categorical", pipe, self.cat_columns)
            )
        # Numerical transformer: min max scalar
        if len(self.num_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "mean")),
                ("onehot", MinMaxScaler())
            ])
            self.column_transformer.transformers.append(            
                ("numerical", pipe, self.num_columns)
            )
        # Set type transformers: multilabel binarizer
        if len(self.set_columns):
            # Each multilabel-column has its own pipeline, because MultilabelBinarizer does not support multi column
            for set_column in self.set_columns: 
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
                    ("binarizer", CustomMultiLabelBinarizer(sparse_output = True))
                ])
                self.column_transformer.transformers.append(
                    (f"set_{set_column}", pipe, [set_column])
                )

        # Perform transformation/preprocessing
        self.df_transform = self.column_transformer.fit_transform(self.df)

        # Return transform data as dataframe if return_dataframe set as True
        if return_dataframe:
            try:
                columns_transform = []
                # Concatenate new transform columns
                if len(self.cat_columns):
                    columns_transform.append(self.column_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_columns))
                if len(self.num_columns):
                    columns_transform.append(numerical_columns)
                if len(self.set_columns):
                    for set_column in self.set_columns: 
                        columns_transform.append(
                            f"{set_column}_" + self.column_transformer.named_transformers_[f"set_{set_column}"].classes_
                        )
                return pd.DataFrame(self.df_transform.toarray(), columns = np.concatenate(columns_transform))
            except MemoryError as e:
                # Memory error. Return array instead
                warnings.warn(f"{str(e)}. Return values instead.")
                return self.df_transform
        else:
            return self.df_transform