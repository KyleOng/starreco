import os
import re
import urllib

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


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


class BaseDataset:
    """
    Base class for dataset.
    """
    
    _download_path = os.path.realpath(__file__).replace("data/dataset/dataset.py", "dataset/")

    def __init__(self):
        # Import data when class object is created
        self.rating, self.user, self.item = self.import_data()

    def download_data(self, url:str):
        """
        Download dataset from the url

        :param url: url to download the dataset
        """
        # Obtain the characters after the last "/" as filename
        file_name = url.split("/")[-1]
        dataset_path = self._download_path + file_name

        # Get file from HTTP request
        response = requests.get(url, stream = True)
        total_url_file_size = int(response.headers.get("content-length", 0))

        # A flag determine whether to download the dataset
        download_flag = False

        # If dataset does not exist or partially download, flag set to True, else False.
        if os.path.isfile(dataset_path):
            total_exist_file_size = int(os.path.getsize(dataset_path)) # Get exist file size

            # Remove partial download dataset
            if total_exist_file_size < total_url_file_size:
                os.remove(dataset_path)
                download_flag = True
        else:
            download_flag = True

        # Download dataset if flag set to True
        if download_flag:
            try:                    
                # Streaming, so we can iterate over the response.
                block_size = 1024 # 1 Kilobyte
                progress_bar = tqdm(total= total_url_file_size, unit="iB", unit_scale = True)

                with open(dataset_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()

                if  total_url_file_size != 0 and progress_bar.n !=  total_url_file_size:
                    raise Exception("Something went wrong during data download.")
            except KeyboardInterrupt:

                # Remove partial download dataset due to keyboard interrupt
                if os.path.isfile(dataset_path):
                    os.remove(dataset_path)
                raise KeyboardInterrupt("Not fully downloaded dataset has been deleted. Please redownload the dataset.")
        
        # Return dataset path
        return dataset_path
        