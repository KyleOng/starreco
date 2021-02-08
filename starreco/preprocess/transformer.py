
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

class Transformer:
    class CustomMultiLabelBinarizer(MultiLabelBinarizer):
        def transform(self, X, y = None):
            return super().transform(X)

        def fit_transform(self, X, y = None):
            return super().fit_transform(X)

    def transform(self, features, return_dataframe = False):

        categorical_columns = features.select_dtypes(["category", "bool"]).columns
        numerical_columns = features.select_dtypes(["int", "float"]).columns
        list_columns = featuress.columns[features.applymap(type).eq(list).any()]

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore"))
        ])

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "mean")),
            ("onehot", MinMaxScaler())
        ])

        list_pipeline = CustomMultiLabelBinarizer()

        transformer = ColumnTransformer([
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns)
            ("list", list_pipeline, list_columns)
        ])

        features_transform = transformer.transform(features)

        if return_dataframe():
        
        else:
            return features_transform