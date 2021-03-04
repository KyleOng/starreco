import tarfile
import warnings
import json

import pandas as pd
import numpy as np

from starreco.data.dataset.dataset import Dataset
from starreco.data.dataset.dataframe import User, Item, Rating

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

        # Parse user trust string to dict
        user_trust_str = tf.extractfile("epinions_data/network_trust.txt").read()
        user_trust_str = user_trust_str.decode("UTF-8")
        user_trust_str = user_trust_str.strip()
        user_trust_dict = user_trust_str.split("\n")
        user_trust_dict = [
            {"user": user_trust.split(" trust ")[0].strip(), 
            "trust": user_trust.split(" trust ")[1].strip()} 
            for user_trust in user_trust_dict
        ]
        # Import user_trust dictionary to dataframe
        user_trust = pd.DataFrame(user_trust_dict) 
        # Get frequent user trust (more than 100)
        # If set all(default), will burn out memory while preprocessing and training
        frequent_trust = user_trust["trust"].value_counts()[user_trust["trust"].value_counts() > 100].keys()
        # Set non-frequent user trust as "others"
        user_trust["frequent_trust"] = np.where(
            user_trust["trust"].isin(frequent_trust),
            user_trust["trust"],
            "others"
        )
        """# Provide another column which store non-frequent user trust
        # Set frequent user trust as NA
        user_trust["others_trust"] = np.where(
            ~user_trust["trust"].isin(frequent_trust),
            user_trust["trust"],
            np.nan
        )"""
        # Apply list (set) on frequent user trust
        user_trust = user_trust.groupby("user")["frequent_trust"].apply(set)\
        .reset_index(name = "frequent_trust")
        # Parse user trustedby string to dict
        user_trustedby_str = tf.extractfile("epinions_data/network_trustedby.txt").read()
        user_trustedby_str = user_trustedby_str.decode("UTF-8")
        user_trustedby_str = user_trustedby_str.strip()
        user_trustedby_dict = user_trustedby_str.split("\n")
        user_trustedby_dict = [
            {"user": user_trustedby.split(" trustedby ")[1].strip(), 
            "trustedby": user_trustedby.split(" trustedby ")[0].strip()} 
            for user_trustedby in user_trustedby_dict
        ]
        # Import user_trustedby dictionary to dataframe
        user_trustedby = pd.DataFrame(user_trustedby_dict)
        # Get frequent user trustedby (more than 100)
        # If set all(default), will burn out memory while preprocessing and training
        frequent_trustedby = user_trustedby["trustedby"].value_counts()[user_trustedby["trustedby"].value_counts() > 100].keys()
        user_trustedby["frequent_trustedby"] = np.where(
            user_trustedby["trustedby"].isin(frequent_trustedby),
            user_trustedby["trustedby"],
            "others"
        )
        """# Provide another column which store non-frequent user trustedby
        # Set frequent user trustedby as NA
        user_trustedby["others_trustedby"] = np.where(
            ~user_trustedby["trustedby"].isin(frequent_trustedby),
            user_trustedby["trustedby"],
            np.nan
        )"""
        # Apply list (set) on frequent user trusedbys
        user_trustedby = user_trustedby.groupby("user")["frequent_trustedby"].apply(set)\
        .reset_index(name = "frequent_trustedby")
        # Merge (inner join) user trust and user trustbys as a single datafarme
        user = user_trust.merge(user_trustedby, on = "user", how = "inner")
        user = User(user, "user")
        user.set_columns = ["frequent_trust", "frequent_trustedby"]

        # Warn user regarding absent of item dataset
        warnings.warn("Epinions dataset does not have items related dataset. Initialize Item with empty dataframe with column instead")
        item = Item(None, "item")

        # Parse ratings string to json
        rating_json_str = tf.extractfile("epinions_data/epinions.json").read()
        rating_json_str = rating_json_str.decode("UTF-8")
        rating_json_str = rating_json_str.strip()
        rating_json_str = "[" + rating_json_str + "]"
        rating_json_str = rating_json_str.replace("\n","")
        rating_json_str = rating_json_str.replace("}{", "},{")
        rating_json = json.loads(json.dumps(eval(rating_json_str)))
        # Normalize ratings json to dataframe
        rating = pd.json_normalize(rating_json)  
        rating = Rating(rating, user, item , "stars")

        return rating, user, item
