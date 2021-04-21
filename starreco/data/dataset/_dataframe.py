import pandas as pd 
import numpy as np
           
class User:
    """
    User dataframe class.

    :param df (pd.DataFrame): User dataframe.

    :param column (str): User column.
    """

    cat_columns = num_columns = set_columns = []

    def __init__(self, df, column):
        self.df = df
        self.column = column


class Item:
    """
    Item dataframe class.

    :param df (pd.DataFrame): Item dataframe.

    :param column (str): Item column.
    """

    cat_columns = num_columns = set_columns = []

    def __init__(self, df, column):
        self.df = df
        self.column = column


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

    def __init__(self, df, column, user = None, item = None):
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
        Return `self.df` with reindexed `self.user.column` and `self.item.column`
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