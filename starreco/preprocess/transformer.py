import warnings

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    def fit_transform(self, X, y = None):
        return super().fit_transform(X)

class Transformer:
    def __init__(self):
        self.categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore"))
        ])

        self.numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "mean")),
            ("onehot", MinMaxScaler())
        ])

        self.list_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
            ("binarizer", MultiColumnMultiLabelBinarizer(dict(sparse_output = True)))
        ])

    def transform(self, df, return_dataframe = False):

        # Auto defined column types. Column types should be set correctly in Dataset module.
        categorical_columns = df.select_dtypes(["category", "bool"]).columns
        numerical_columns = df.select_dtypes(["int", "float"]).columns
        list_columns = df.columns[df.applymap(type).eq(list).any()]

        transformer = ColumnTransformer([
            ("categorical", self.categorical_pipeline, categorical_columns),
            ("numerical", self.numerical_pipeline, numerical_columns),
            ("list", self.list_pipeline, list_columns)
        ])

        df_transform = transformer.fit_transform(df)

        if return_dataframe:
            try:
                columns_transform = []
                if len(categorical_columns):
                    columns_transform.append(transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_columns))
                if len(numerical_columns):
                    columns_transform.append(numerical_columns)
                if len(list_columns):
                    columns_transform.append(transformer.named_transformers_["list"].classes_)
                return pd.DataFrame(df_transform.toarray(), columns = np.concatenate(columns_transform))
            except MemoryError as e:
                warnings.warn(f"{str(e)}. Return values instead.")
                return df_transform
            except Exception:
                warnings.warn("Dataframe empty.")
                return df_transform
        else:
            return df_transform