import urllib
import os
import json
import re
import warnings
import zipfile, tarfile

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

class Dataset:
    datasets_path = "starreco/dataset/"
    ratings = None
    users = None
    items = None
    rated_users = None
    rated_items = None
    
    def __init__(self):
        pass

    def download_data(self, url:str):
        """
        Download dataset from the url
        """
        # Obtain the characters after the last "/" as filename
        file_name = url.split("/")[-1]
        dataset_path = self.datasets_path + file_name

        # Get file from HTTP request
        response = requests.get(url, stream=True)
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
                progress_bar = tqdm(total= total_url_file_size, unit="iB", unit_scale=True)

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

    def prepare_data(self):
        # Import data
        ratings, users, items = self.import_data()

        # Rating dataset only focused on three attributes - user, item and rating
        ratings = ratings[[self.user_column, self.item_column, self.rating_column]]

        # If users dataset exist, factorize user_id in rating dataset and map to user dataset
        if users is not None:
            # user_id_code = Factorized user_id
            # user_id_reversed = Original user_id

            ratings = ratings.rename({self.user_column: self.user_column + "_code"}, axis = 1)
            users = users.rename({self.user_column: self.user_column + "_reversed"}, axis = 1)

            ratings[self.user_column + "_code"] = ratings[self.user_column + "_code"].astype("category")
            users[self.user_column + "_reversed"] = users[self.user_column + "_reversed"].astype("category")

            user_maps = dict(enumerate(ratings[self.user_column + "_code"].cat.categories))
            ratings[self.user_column + "_code"] = ratings[self.user_column + "_code"].cat.codes
            users[self.user_column + "_code"] = users[self.user_column + "_reversed"].map(
                dict((y,x) for x,y in user_maps.items())
            )
            rated_users = ratings.merge(users, how = "left", on = self.user_column + "_code")[users.columns]

            # Set users and rated_users as class property
            self.user_column = self.user_column + "_code"
            self.users = users
            self.rated_users = rated_users.loc[:, ~rated_users.columns.str.contains(f"^{self.user_column}", case=False)] 

        # If items dataset exist, factorize item_id in ratings dataset and map to items dataset
        if items is not None:
            # item_id_code = Factorized item_id
            # item_id_reversed = Original item_id

            ratings = ratings.rename({self.item_column: self.item_column + "_code"}, axis = 1)
            items = items.rename({self.item_column: self.item_column + "_reversed"}, axis = 1)

            ratings[self.item_column + "_code"] = ratings[self.item_column + "_code"].astype("category")
            items[self.item_column + "_reversed"] = items[self.item_column + "_reversed"].astype("category")

            item_maps = dict(enumerate(ratings[self.item_column + "_code"].cat.categories))
            ratings[self.item_column + "_code"] = ratings[self.item_column + "_code"].cat.codes
            items[self.item_column + "_code"] = items[self.item_column + "_reversed"].map(
                dict((y,x) for x,y in item_maps.items())
            )
            rated_items = ratings.merge(items, how = "left", on = self.item_column + "_code")[items.columns]

            # Set items and rated_items as class property
            self.item_column = self.item_column + "_code"
            self.items = items
            self.rated_items = rated_items.loc[:, ~rated_items.columns.str.contains(f"^{self.item_column}", case=False)] 

        # Set ratings as class property
        self.ratings = ratings
        return ratings

class MovielensDataset(Dataset):
    rating_column = "rating"
    user_column = "userId"
    item_column = "movieId"

    def __init__(self, size = "1m"):
        super().__init__()
        self.size = size # ** haven't implement for various movielens dataset size **
            
    def import_data(self):
        """
        Import Movielens dataset
        ~ Movielens also provides latest dataset, but only suitable for development and eduction.
        ~ Latest Movielens dataset also not appropriate for reporting research results.
        ~ Hence, old datasets are utilized for benchmarking.
        ~ For more info: https://grouplens.org/datasets/movielens/
        """
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        ratings = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
        names = ["userId", "movieId", "rating", "timestamp"], engine = "python")

        users = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::",
        names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        users[["gender", "occupation", "zipCode"]] = users[["gender", "occupation", "zipCode"]].astype("category")

        items = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::",
        names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        items["genre"] = items["genre"].apply(lambda x:x.split("|"))
        items[["title"]] = items[["title"]].astype("category")

        return ratings, users, items
        
class EpinionsDataset(Dataset):
    rating_column = "stars"
    user_column = "user"
    item_column = "item"

    def import_data(self):
        """
        Import Epinions Dataset
        ~ Json parsing is required as Epinions does not provide clean json format datasets.
        """
        # Get dataset
        dataset_path = super().download_data("http://deepyeti.ucsd.edu/jmcauley/datasets/epinions/epinions_data.tar.gz")
        tf = tarfile.open(dataset_path , "r:gz") 

        # Parse ratings string to json
        ratings_json_str = tf.extractfile("epinions_data/epinions.json").read()
        ratings_json_str = ratings_json_str.decode("UTF-8")
        ratings_json_str = ratings_json_str.strip()
        ratings_json_str = "[" + ratings_json_str + "]"
        ratings_json_str = ratings_json_str.replace("\n","")
        ratings_json_str = ratings_json_str.replace("}{", "},{")
        ratings_json_parse = json.loads(json.dumps(eval(ratings_json_str)))
        ratings = pd.json_normalize(ratings_json_parse)

        # Parse users trusts string to dict
        user_trusts_str = tf.extractfile("epinions_data/network_trust.txt").read()
        user_trusts_str = user_trusts_str.decode("UTF-8")
        user_trusts_str = user_trusts_str.strip()
        user_trusts_parse = user_trusts_str.split("\n")
        user_trusts_parse = [
            {"user": user_trust.split(" trust ")[0].strip(), 
            "trust": user_trust.split(" trust ")[1].strip()} 
            for user_trust in user_trusts_parse
        ]
        user_trusts = pd.DataFrame(user_trusts_parse)
        user_trusts = user_trusts.groupby("user")["trust"].apply(list).reset_index(name = "trusts")

        # Parse users trustbys string to dict
        user_trustedbys_str = tf.extractfile("epinions_data/network_trustedby.txt").read()
        user_trustedbys_str = user_trustedbys_str.decode("UTF-8")
        user_trustedbys_str = user_trustedbys_str.strip()
        user_trustedbys_parse = user_trustedbys_str.split("\n")
        user_trustedbys_parse = [
            {"user": user_trustedby.split(" trustedby ")[1].strip(), 
            "trustedby": user_trustedby.split(" trustedby ")[0].strip()} 
            for user_trustedby in user_trustedbys_parse
        ]
        user_trustedbys = pd.DataFrame(user_trustedbys_parse)
        user_trustedbys = user_trustedbys.groupby("user")["trustedby"].apply(list).reset_index(name = "trustedby")

        users = user_trusts.merge(user_trustedbys, on = "user")

        # Merge (inner join) two users datafarme
        # users = user_trusts.merge(user_trustedbys, on = "user")

        # Warn users regarding absent of item dataset
        # warnings.warn("Epinions dataset does not have items related dataset. Return None instead")

        return ratings, users, None

class BookCrossingDataset(Dataset):
    rating_column = "Book-Rating"
    user_column = "User-ID"
    item_column = "ISBN"

    def import_data(self):
        """
        Import Book Crossing Dataset
        """
        # Get dataset
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)

        ratings = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")

        users = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        users["Location"] = users["Location"].apply(lambda x:x.split(", "))

        items = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        items[["Book-Author", "Publisher"]] = items[["Book-Author", "Publisher"]].astype("category")
        items = items[["ISBN", "Book-Author", "Publisher"]]

        return ratings, users, items
    