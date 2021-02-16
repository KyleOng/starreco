import warnings

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    def fit_transform(self, X, y = None):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype = "object")
        return super().fit_transform(X.flatten())

class Preprocessor:
    def df_to_sparse(self, df, user_column, item_column):
        """
        Transform user-item rating dataframe into sparse matrix.
        :param df: ratings dataframe.
        :param user_column: user column name.
        :param item_column: item column name.
        """
        num_users = df[user_column].nunique()
        num_items = df[item_column].nunique()

        matrix = lil_matrix((num_users, num_items), dtype = "uint8")
        for _, user, item, rating in df.itertuples():
            matrix[user, item] = rating

        #breakpoint() # Evaluate matrix
        #[[e,i] for e, i in enumerate(matrix[0].toarray()[1]) if i > 0]

        return matrix            

    def transform(self, df, return_dataframe = False):
        """
        Transform dataframe into model compatible format.
        :param df: pandas dataframe.
        :param return_dataframe: return dataframe if True, else return sparse matrix. 
        :return: dataframe if return_dataframe set as True, else return sparse matrix.
        """
        # Auto defined column types. Column types should be set correctly in Dataset module.
        categorical_columns = df.select_dtypes(["category", "bool"]).columns
        numerical_columns = df.select_dtypes(["int", "float"]).columns
        list_columns = df.columns[df.applymap(type).eq(set).any()]

        # Return empty array if dataframe is empty
        if len(categorical_columns) == 0 and len(numerical_columns) == 0 \
        and len(list_columns) == 0:
            return []

        # Dynamic transformer/pipeline construction
        column_transformer = ColumnTransformer([])
        # Each column type has its own transformation pipeline
        # Categorical transformer: one hot encoder
        if len(categorical_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ])
            column_transformer.transformers.append(
                ("categorical", pipe, categorical_columns)
            )
        # Numerical transformer: min max scalar
        if len(numerical_columns):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "mean")),
                ("onehot", MinMaxScaler())
            ])
            column_transformer.transformers.append(            
                ("numerical", pipe, numerical_columns)
            )
        # List type transformers: multilabel binarizer
        if len(list_columns):
            # Each column has its own pipeline, because MultilabelBinarizer does not support multi column
            for list_column in list_columns: 
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
                    ("binarizer", CustomMultiLabelBinarizer(sparse_output = True))
                ])
                column_transformer.transformers.append(
                    (f"list_{list_column}", pipe, [list_column])
                )

        # Perform transformation/preprocessing
        df_transform = column_transformer.fit_transform(df)

        # Return transform data as dataframe if return_dataframe set as True
        if return_dataframe:
            try:
                columns_transform = []
                # Concatenate new transform columns
                if len(categorical_columns):
                    columns_transform.append(column_transformer.named_transformers_["categorical"].named_steps["onehot"].get_feature_names(categorical_columns))
                if len(numerical_columns):
                    columns_transform.append(numerical_columns)
                if len(list_columns):
                    for list_column in list_columns: 
                        columns_transform.append(
                            f"{list_column}_" + column_transformer.named_transformers_[f"list_{list_column}"].classes_
                        )
                return pd.DataFrame(df_transform.toarray(), columns = np.concatenate(columns_transform))
            except MemoryError as e:
                # Memory error. Return array instead
                warnings.warn(f"{str(e)}. Return values instead.")
                return df_transform
        else:
            return df_transform