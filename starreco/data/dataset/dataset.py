import os

import pandas as pd
import requests
from tqdm import tqdm

# Done
class User:
    """
    User dataframe class.

    - df (pd.DataFrame): User dataframe.
    - column (str): User column.
    """

    cat_columns = num_columns = set_columns = doc_columns = []

    def __init__(self, df:pd.DataFrame, column:str):
        self.df = df
        self.column = column

# Done
class Item:
    """
    Item dataframe class.

    - df (pd.DataFrame): Item dataframe.
    - column (str): Item column.
    """

    cat_columns = num_columns = set_columns = doc_columns = []

    def __init__(self, df:pd.DataFrame, column:str):
        self.df = df
        self.column = column

# Done
class Rating:
    """
    Rating dataframe class.

    - df (pd.DataFrame): Rating dataframe.
    - column (str): Rating column.
    - user (User): User dataframe class.
    - item (Item): Item dataframe class.
    """

    def __init__(self, 
                 df:pd.DataFrame, 
                 column:str, 
                 user = None, 
                 item = None):
        self.df = df
        self.user = user
        self.item = item
        self.column = column
        self.num_users = df[user.column].nunique()
        self.num_items = df[item.column].nunique()

    def clean(self):
        """
        data cleaning.
        """
        df = self.df.copy()
        user_df = self.user.df.copy()
        item_df = self.item.df.copy()

        user_df = user_df.dropna()
        item_df = item_df.dropna()

        df = df.merge(user_df, on = self.user.column).merge(item_df, on = self.item.column)[df.columns]
        user_df = user_df.merge(df[self.user.column].drop_duplicates(), on = self.user.column)
        item_df = item_df.merge(df[self.item.column].drop_duplicates(), on = self.item.column)
        

        return df, user_df, item_df

# Done
class BaseDataset:
    """
    Base class for dataset.
    """

    def __init__(self, crawl_data:bool = True):
        self.crawl_data = crawl_data
        # Import data when class object is created
        self.rating, self.user, self.item = self.import_data()

    def download_data(self, url):
        """
        Download dataset from the url

        - url: url to download the dataset

        - return: dataset path.
        """
        # Obtain the characters after the last "/" as filename
        dataset_path = url.split("/")[-1]

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
                progress_bar = tqdm(total = total_url_file_size, unit = "iB", unit_scale = True)

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