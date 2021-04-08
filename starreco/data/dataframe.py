import pandas as pd 
import numpy as np

class DataframeMixin:
    """
    Mixin class for pandas dataframe.

    :param df (pd.DataFrame): Dataframe.

    :param column (str): Column name.

    Warning: This class should not be used directly. Use derived class instead.
    """

    cat_columns = num_columns = set_columns = []

    def __init__(self, df:pd.DataFrame, column:str):
        self.df = df if df is not None else pd.DataFrame(columns = [column])
        self.column = column
        
    def map_column(self, map:dict, join:str = "left"):
        """
        Mapping operation.

        :param map (dict): A dictionary which store mapping values which maps to the `self.df[self.column]`

        :join (str): Type of table join to be performed between `self.df` and `df_map` 一 a dataframe created from `map`. Table join will be performed if `df[self.column].values` do not store the same shape and element as `map.values()`. Table join will also be disregard if set as None. Default: "left"
        """

        df = self.df.copy()
        self.map_ = map
        df[self.column] = df[self.column].map(map)
        if join:
            # Perform table join if `df[self.column].values` do not store the same shape and element as `map.values()`
            if not np.array_equal(df[self.column].values, list(map.values())):
                # Create a dataframe from `map`
                df_map = pd.DataFrame(map.values(), columns = [self.column])
                # Table join between `df_map` and `self.df`
                df = df_map.merge(df, on = self.column, how = join)
        return df.sort_values(self.column)
            
class User(DataframeMixin):
    """
    User dataframe class.

    :param df (pd.DataFrame): User dataframe.

    :param column (str): User column.
    """

    def __init__(self, df, column):
        super().__init__(df, column)
        
class Item(DataframeMixin):
    """
    Item dataframe class.

    :param df (pd.DataFrame): Item dataframe.

    :param column (str): Item column.
    """

    def __init__(self, df, column):
        super().__init__(df, column)
        
class Rating:
    """
    Rating dataframe class.

    :param df (pd.DataFrame): Rating dataframe.

    :param user (User): User dataframe class.

    :param item (Item): Item dataframe class.

    :param column (str): Rating column.

    :param num_users (int): Number of users.

    :param num_items (int): Number of items.
    """

    def __init__(self, df, user, item, column):
        self.df = df
        self.user = user
        self.item = item
        self.column = column
        self.num_users = df[user.column].nunique()
        self.num_items = df[item.column].nunique()
        
    def _reindex(self, df, column):
        """
        Reindex a dataframe column.

        :param df (pd.DataFrame): Dataframe.

        :param column (str): Column.

        :return df: reindexed df.
        
        Warning: This method should not be used directly.
        """

        df = df.copy()
        df[column] = df[column].astype("category").cat.codes
        return df
    
    @property
    def reindex(self):
        """
        Return reindexed `self.df` on `self.user.column` and `self.item.column`
        """

        return self._reindex(
            self._reindex(self.df, self.user.column), 
            self.item.column
        )
    
    @property
    def user_map(self):
        """
        Return a map dictionary which store reindexed user values.
        """

        return {v: i for i, v in
        enumerate(self.df[self.user.column].astype("category").cat.categories)}
        
    @property
    def item_map(self):
        """
        Return a map dictionary which store reindexed item values.
        """

        return {v: i for i, v in
        enumerate(self.df[self.item.column].astype("category").cat.categories)}
    
    def _select_related(self, parent_df, column):
        """
        Return a dataframe which "follow" the foreign-key relationship on `column` between parent class 一 `parent_df` and foreign class 一 `self.df`. 

        Warning: This method should not be used directly.
        """

        merge = self.df.merge(parent_df, on = column, how = "left")
        return merge[parent_df.columns]
    
    @property
    def user_select_related(self):
        """
        Return a dataframe which "follow" the foreign-key relationship on `self.user.column` between parent class 一 `self.user.df` and foreign class 一 `self.df`.
        """

        return self._select_related(self.user.df, self.user.column)
    
    @property
    def item_select_related(self):
        """
        Return a dataframe which "follow" the foreign-key relationship on `self.item.column` between parent class 一 `self.item.df` and foreign class 一 `self.df`. 
        """

        return self._select_related(self.item.df, self.item.column)