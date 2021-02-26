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

    def preprocessing(self, ratings, objects, column):
        factorize_column = column + "_fac"
        original_column = column + "_ori"

        ratings = ratings.rename({column: factorize_column}, axis = 1)

        ratings[factorize_column] = ratings[factorize_column].astype("category")
        categories = ratings[factorize_column].cat.categories
        ratings[factorize_column] = ratings[factorize_column].cat.codes

        if objects is not None:
            objects = objects.rename({column: original_column}, axis = 1)

            objects[original_column] = objects[original_column].astype("category")
            objects[factorize_column] = objects[original_column].map(
                dict((original, factorize)
                for factorize, original in dict(enumerate(categories)).items())
            )

            if not set(ratings[factorize_column]).issubset(set(objects[factorize_column])):
                objects = pd.DataFrame({factorize_column: [i for i in range(len(categories))]})\
                .merge(objects, on = factorize_column, how = "outer")

            rated_objects = ratings.merge(objects, on = factorize_column, how = "left")[objects.columns]
            rated_objects = rated_objects.loc[:, ~rated_objects.columns.str.contains(f"^{column}", case = False)]

            return ratings, objects, rated_objects
        else:
            objects = pd.DataFrame(
                {factorize_column: [i for i in range(len(categories))], 
                original_column: categories}
            )

            return ratings, objects, None

    def __init__(self):
        # Import data
        ratings, users, items = self.import_data()

        # Rating dataset only focused on three attributes - user, item and rating
        ratings = ratings[[self.user_column, self.item_column, self.rating_column]]

        self.num_users = ratings[self.user_column].nunique()
        self.num_items = ratings[self.item_column].nunique()

        ratings, self.users, self.rated_users = self.preprocessing(ratings, users, self.user_column)
        self.ratings, self.items, self.rated_items = self.preprocessing(ratings, items, self.item_column)
        
        self.user_column = list(set(self.ratings.columns) & set(self.users.columns))[0]
        self.item_column = list(set(self.ratings.columns) & set(self.items.columns))[0]

class MovielensDataset(Dataset):
    rating_column = "rating"
    user_column = "userId"
    item_column = "movieId"
            
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
        items["genre"] = items["genre"].apply(lambda x:set(x.split("|")))

        return ratings, users, items

    def __init__(self, size = "1m"):
        self.size = size # ** haven't implement for various movielens dataset size **
        super().__init__()
        
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
        ratings_json = json.loads(json.dumps(eval(ratings_json_str)))
        # Normalize ratings json to dataframe
        ratings = pd.json_normalize(ratings_json)  

        # Parse users trusts string to dict
        user_trusts_str = tf.extractfile("epinions_data/network_trust.txt").read()
        user_trusts_str = user_trusts_str.decode("UTF-8")
        user_trusts_str = user_trusts_str.strip()
        user_trusts_dict = user_trusts_str.split("\n")
        user_trusts_dict = [
            {"user": user_trust.split(" trust ")[0].strip(), 
            "trust": user_trust.split(" trust ")[1].strip()} 
            for user_trust in user_trusts_dict
        ]
        # Import user_trusts dictionary to dataframe
        user_trusts = pd.DataFrame(user_trusts_dict) 
        self.user_trusts = user_trusts
        # Get frequent user trusts (more than 100)
        # If set all(default), will burn out memory while preprocessing and training
        frequent_trusts = user_trusts["trust"].value_counts()[user_trusts["trust"].value_counts() > 100].keys()
        # Set non-frequent user trusts as "others"
        user_trusts["frequent_trust"] = np.where(
            user_trusts["trust"].isin(frequent_trusts),
            user_trusts["trust"],
            "others"
        )
        # Provide another column which store non-frequent user trusts
        # Set frequent user trusts as NA
        user_trusts["others_trust"] = np.where(
            ~user_trusts["trust"].isin(frequent_trusts),
            user_trusts["trust"],
            np.nan
        )
        # Apply list (set) on frequent user trusts
        user_trusts = user_trusts.groupby("user")["frequent_trust"].apply(set)\
        .reset_index(name = "frequent_trust")
        
        # Parse users trustedbys string to dict
        user_trustedbys_str = tf.extractfile("epinions_data/network_trustedby.txt").read()
        user_trustedbys_str = user_trustedbys_str.decode("UTF-8")
        user_trustedbys_str = user_trustedbys_str.strip()
        user_trustedbys_dict = user_trustedbys_str.split("\n")
        user_trustedbys_dict = [
            {"user": user_trustedby.split(" trustedby ")[1].strip(), 
            "trustedby": user_trustedby.split(" trustedby ")[0].strip()} 
            for user_trustedby in user_trustedbys_dict
        ]
        # Import user_trustedbys dictionary to dataframe
        user_trustedbys = pd.DataFrame(user_trustedbys_dict)
        self.user_trustedbys = user_trustedbys
        # Get frequent user trustedbys (more than 100)
        # If set all(default), will burn out memory while preprocessing and training
        frequent_trustedbys = user_trustedbys["trustedby"].value_counts()[user_trustedbys["trustedby"].value_counts() > 100].keys()
        user_trustedbys["frequent_trustedby"] = np.where(
            user_trustedbys["trustedby"].isin(frequent_trustedbys),
            user_trustedbys["trustedby"],
            "others"
        )
        # Provide another column which store non-frequent user trustedbys
        # Set frequent user trustedbys as NA
        user_trustedbys["others_trustedby"] = np.where(
            ~user_trustedbys["trustedby"].isin(frequent_trustedbys),
            user_trustedbys["trustedby"],
            np.nan
        )
        # Apply list (set) on frequent user trusedbys
        user_trustedbys = user_trustedbys.groupby("user")["frequent_trustedby"].apply(set)\
        .reset_index(name = "frequent_trustedby")
        
        # Merge (inner join) user trusts and user trustbys as a single datafarme
        users = user_trusts.merge(user_trustedbys, on = "user", how = "inner")
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
        users["Location"] = users["Location"].astype("category")

        items = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        items[["Book-Author", "Publisher"]] = items[["Book-Author", "Publisher"]].astype("category")
        items = items[["ISBN", "Book-Author", "Publisher"]]

        return ratings, users, items
    