import urllib
import os
import re
import zipfile

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from .dataframe import User, Item, Rating

class BaseDataset:
    """
    Base class for dataset.
    """

    _datasets_path = os.path.realpath(__file__).replace("data/dataset.py", "dataset/")

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
        dataset_path = self._datasets_path + file_name

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


class MovielensDataset(BaseDataset):
    """
    Dataset class for Movielens.
    """

    def import_data(self):
        """
        Import Movielens rating, user and item dataframes. 
        """

        # Download dataset from url and store as zipfile
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        user = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::",
                           names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        user = User(user, "userId")
        user.cat_columns = ["gender", "occupation", "age"]

        # Item dataframe
        item = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::",
                           names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        item["genre"] = item["genre"].apply(lambda x:set(x.split("|")))
        item = Item(item, "movieId")
        item.set_columns = ["genre"]

        # Rating dataframe
        rating = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
                             names = ["userId", "movieId", "rating", "timestamp"], engine = "python")
        rating = Rating(rating, user, item , "rating")

        return rating, user, item


class BookCrossingDataset(BaseDataset):
    """
    Dataset class for Book Crossing.
    """

    def import_data(self):
        """
        Import Book Crossing rating, user and item dataframes. 
        """

        # Download dataset from url and store as zipfile
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        user = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        user = User(user, "User-ID")
        user.cat_columns = ["Location"]
        user.num_columns = ["Age"]

        # Item dataframe
        item = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        item = Item(item, "ISBN")
        item.cat_columns = ["Book-Author", "Publisher"]

        # Rating dataframe
        rating = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        rating = Rating(rating, user, item, "Book-Rating")

        return rating, user, item

