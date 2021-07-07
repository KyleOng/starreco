import warnings

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse.csr import csr_matrix

from .transformers import CustomMultiColumnLabelEncoder, SetTransformer, DocTransformer

# Done
class Preprocessor:
    """
    Preprocessor class for data prerocessing.

    cat_columns (list): List of categorical columns. Default = [].
    num_columns (list): List of numerical columns. Default = [].
    set_columns (list): List of set columns. Default = [].
    doc_columns (list): List of document columns. Default = [].
    cat_transformer (str): Column transformer for categorical columns, either ["onehot", "label"]. Default: onehot.
    """
    
    def __init__(self, 
                 df:pd.DataFrame, 
                 cat_columns:list = [], 
                 num_columns:list = [], 
                 set_columns:list = [],
                 doc_columns:list = [],
                 cat_transformer:str = "label"): 
        assert cat_transformer in ["onehot", "label"], "cat_transformer can be either 'onehot' (OneHotEncoder) or 'label' (LabelEncoder)"

        self.df = df
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.set_columns = set_columns
        self.doc_columns = doc_columns
        self.cat_transformer = cat_transformer
        #print(self.cat_columns, self.num_columns, self.set_columns, self.doc_columns)

    def get_feature_names(self):
        columns_transform = {"categorical": [],
                             "numerical": [],
                             "set": [],
                             "document": []}

        if self.cat_columns:
            if self.cat_transformer == "onehot":
                columns_transform["categorical"] = self.column_transformer.named_transformers_["categorical"]\
                                                .named_steps[self.cat_transformer].get_feature_names(self.cat_columns)
            elif self.cat_transformer == "label":
                columns_transform["categorical"] = self.cat_columns
        if self.num_columns:
            columns_transform["numerical"] = self.num_columns
        if self.set_columns:
            columns_transform["set"] = self.column_transformer.named_transformers_["set"].get_feature_names()
        if self.doc_columns:
            for name, max_len in self.column_transformer.named_transformers_["document"].max_lens_.items():
                for i in range(max_len):
                    columns_transform["document"].append(f"{name}_{i}")

        return columns_transform

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
        self.column_transformer = ColumnTransformer([], sparse_threshold = 0.5)

        # Each column type has its own transformation pipeline
        # Categorical transformer: one hot encoder
        if self.cat_columns:
            if self.cat_transformer == "onehot":
                cat_transformer = OneHotEncoder(handle_unknown = "ignore")
            elif self.cat_transformer == "label":
                cat_transformer = CustomMultiColumnLabelEncoder()
            pipe = Pipeline([("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                             (self.cat_transformer, cat_transformer)])
            self.column_transformer.transformers.append(("categorical", pipe, self.cat_columns))

        # Numerical transformer: min max scalar
        if self.num_columns:
            pipe = Pipeline([("imputer", SimpleImputer(strategy = "mean")),
                             ("onehot", MinMaxScaler())])
            self.column_transformer.transformers.append(("numerical", pipe, self.num_columns))

        # Set type transformers: multilabel binarizer
        if self.set_columns:
            # Each multilabel-column has its own pipeline, because MultilabelBinarizer does not support multi column
           set_transformer = SetTransformer()
           self.column_transformer.transformers.append(("set", set_transformer, self.set_columns))

        # Doc type transformers: word indixer based on glove pretrained weights
        if self.doc_columns:
            warnings.warn(f"Doc type columns exist. Make sure 'glove.6B/glove.6B.50d.txt' is in your working directory." )
            doc_transformer = DocTransformer()
            self.column_transformer.transformers.append(("document", doc_transformer, self.doc_columns))

        # Perform transformation/preprocessing
        self.df_transform = self.column_transformer.fit_transform(self.df)

        # Return transform data as dataframe if return_dataframe set as True
        if return_dataframe:
            try:
                columns_transform = np.concatenate(list(self.get_feature_names().values()))    
                if isinstance(self.df_transform, csr_matrix):  
                    return pd.DataFrame(self.df_transform.toarray(), columns = columns_transform)
                else:
                    return pd.DataFrame(self.df_transform, columns = columns_transform)
            except MemoryError as e:
                # Memory error. Return array instead
                warnings.warn(f"{str(e)}. Return values instead.")
                return self.df_transform
        else:
            return self.df_transform