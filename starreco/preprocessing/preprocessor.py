import warnings

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .transformers import SetTransformer, DocTransformer

# Done
class Preprocessor:
    """
    Preprocessor class for data prerocessing.

    cat_columns (list): List of categorical columns.
    num_columns (list): List of numerical columns.
    set_columns (list): List of set columns.
    doc_columns (list): List of document columns.
    """
    
    def __init__(self, 
                 df:pd.DataFrame, 
                 cat_columns:list = [], 
                 num_columns:list = [], 
                 set_columns:list = [],
                 doc_columns:list = []):
        self.df = df
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.set_columns = set_columns
        self.doc_columns = doc_columns
        print(self.cat_columns, self.num_columns, self.set_columns, self.doc_columns)

    def transform(self, return_dataframe = False):
        """
        Transform dataframe into algorithm compatible format.

        :return: dataframe if return_dataframe set as True, else return sparse matrix. Default: False
        """

        # Return None if dataframe is empty
        if self.df is None:
            return None

        # Return None if cat/num/self.set_columns are empty
        if bool(self.cat_columns) and bool(self.num_columns) and bool(self.set_columns) and bool(self.doc_columns):
            return None

        # Dynamic transformer/pipeline construction
        self.column_transformer = ColumnTransformer([])

        # Each column type has its own transformation pipeline
        # Categorical transformer: one hot encoder
        if self.cat_columns:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ])
            self.column_transformer.transformers.append(
                ("categorical", pipe, self.cat_columns)
            )

        # Numerical transformer: min max scalar
        if self.num_columns:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "mean")),
                ("onehot", MinMaxScaler())
            ])
            self.column_transformer.transformers.append(            
                ("numerical", pipe, self.num_columns)
            )

        # Set type transformers: multilabel binarizer
        if self.set_columns:
            # Each multilabel-column has its own pipeline, because MultilabelBinarizer does not support multi column
           set_transformer = SetTransformer()
           self.column_transformer.transformers.append(
                ("set", set_transformer, self.set_columns)
            )

        # Doc type transformers: tfidf vectorizer
        if self.doc_columns:
            doc_transformer = DocTransformer()
            self.column_transformer.transformers.append(
                ("document", doc_transformer, self.doc_columns)
            )

        # Perform transformation/preprocessing
        self.df_transform = self.column_transformer.fit_transform(self.df)

        # Return transform data as dataframe if return_dataframe set as True
        if return_dataframe:
            try:
                columns_transform = []
                # Concatenate new transform columns
                if self.cat_columns:
                    columns_transform.append(self.column_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(self.cat_columns))
                if self.num_columns:
                    columns_transform.append(self.num_columns)
                if self.set_columns:
                    columns_transform.append(self.column_transformer.named_transformers_[f"set"].get_feature_names())
                if self.doc_columns:
                    columns_transform.append(self.column_transformer.named_transformers_[f"document"].get_feature_names())
                return pd.DataFrame(self.df_transform.toarray(), columns = np.concatenate(columns_transform))
            except MemoryError as e:
                # Memory error. Return array instead
                warnings.warn(f"{str(e)}. Return values instead.")
                return self.df_transform
        else:
            return self.df_transform