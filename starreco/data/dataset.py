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
        :param url: url to download the dataset
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

    def _reindex(self, ratings:pd.DataFrame, column:str, map_objects:bool = False, 
                objects:pd.DataFrame = None, outer_join:bool = False):
        """
        Reindex/Factorize certain column of the dataframe
        :param ratings: Ratings dataframe.
        :param column: Column which to be reindexed/factorized.        
        :param map_objects: Map the reindexed column from the ratings dataframe to the original column of the object dataframe if set to True, else abort.
        :param objects: Objects (users or items) dataframe which one of its column is reference to the ratings dataframe.
        :param outer_join: Perform outer join between ratings and objects dataframe, as the final objects dataframe.
        :return: Ratings and objects dataframes with reindexed columns if map_objects set to True, else ratings dataframe with reindexed columns only.
        """
        # Column to be reindexed/factorized
        factorize_column = column
        ratings = ratings.rename({column: factorize_column}, axis = 1) # Rename column
        

        # Change dtype to category
        ratings[factorize_column] = ratings[factorize_column].astype("category")
        # Store original values as respect to reindex values.
        categories = ratings[factorize_column].cat.categories
        # Reindex column in the ratings dataset.
        ratings[factorize_column] = ratings[factorize_column].cat.codes
        # Mapping objects dataframe.
        if map_objects:
            # Original column will be mapped with reindex values.
            # Original values will be stored in "_ori" column
            original_column = column + "_ori"

            # Perform mapping if objects dataframe exist.
            # else return objects dataframe that only maps reindex values to original values.
            if objects is not None:
                # Rename column by adding "_ori".
                # This column will store the original values.
                objects = objects.rename({column: original_column}, axis = 1)
                # Change dtype to category.
                objects[original_column] = objects[original_column].astype("category") 
                # Map the reindexed column from the ratings dataframe, to the original column of the object dataframe.
                objects[factorize_column] = objects[original_column].map(
                    dict((original, factorize)
                    for factorize, original in dict(enumerate(categories)).items())
                )

                if outer_join:
                    # Perform outer join if the not all column values in the ratings dataframe in objects dataframe.
                    # Outer join makes sure that all the column values exist in ratings dataframe, also exist in objects dataframe.
                    if not np.isin(ratings[factorize_column].unique(), 
                    objects[factorize_column].unique()).all():
                        objects = pd.DataFrame({factorize_column: [i for i in range(len(categories))]})\
                        .merge(objects, on = factorize_column, how = "outer")
            else:
                # Create and return objects dataframe which store original values as respect to reindex values.
                objects = pd.DataFrame(
                    {factorize_column: [i for i in range(len(categories))], 
                    original_column: categories}
                )

            # Sort columns
            objects = objects.reindex([factorize_column, original_column] + 
                                    [col for col in objects.columns 
                                    if col not in [factorize_column, original_column]], 
                                    axis=1)

            objects = objects.sort_values(column)
            return ratings, objects
        else:
            return ratings
    
    def _merge_ratings(self, ratings:pd.DataFrame, objects:pd.DataFrame, column:str, drop_left_columns:bool = True):
        if objects is not None:
            rated_objects = ratings.merge(objects, on = column, how = "left")
            if drop_left_columns:
                rated_objects = rated_objects.drop(ratings.columns, axis = 1)
            return rated_objects
        else: 
            return None

    def get_ratings(self):
        return self._reindex(self._reindex(self.ratings_, self.user_column), 
        self.item_column)[[self.user_column, self.item_column, self.rating_column]]

    def get_users(self, merge_ratings = False, features_only = True):
        if merge_ratings:
            if features_only:   
                return self._merge_ratings(self.ratings_, self.users_, self.user_column)
            else:
                return self._merge_ratings(self.ratings_, self.users_, self.user_column, 
                drop_left_columns = False)
        else:
            _, users = self._reindex(self.ratings_, self.user_column, map_objects = True, 
            objects = self.users_, outer_join = True)
            if features_only:  
                return users.dropna(subset = [self.user_column]).iloc[:, 2:]
            else:
                return users

    def get_items(self, merge_ratings = False, features_only = True):
        if merge_ratings:
            if features_only:   
                return self._merge_ratings(self.ratings_, self.items_, self.item_column)
            else:
                return self._merge_ratings(self.ratings_, self.items_, self.item_column, 
                drop_left_columns = False)
        else:
            _, items = self._reindex(self.ratings_, self.item_column, map_objects = True, 
            objects = self.items_, outer_join = True)
            if features_only:  
                return items.dropna(subset = [self.item_column]).iloc[:, 2:]
            else:
                return items

    def __init__(self):
        # Import data
        self.ratings_, self.users_, self.items_ = self.import_data()

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
        items = items.drop(["title"], axis = 1)

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
    