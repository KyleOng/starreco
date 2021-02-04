import urllib
import os
import json
import re
import zipfile, tarfile

from tqdm import tqdm
import pandas as pd
import requests

class Dataset:
    datasets_path = "starreco/dataset/"
    
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

class MovielensDataset(Dataset):
    def __init__(self, type = "1m"):
        super().__init__()
        self.type = type
            
    def import_data(self):
        """
        Import Movielens dataset
        1. Get dataset path
        2. Extract dataset from archive file
        3. Dataset to dataframe
        """
        # old dataset
        if self.type == "1m":
            dataset_path = super().download_data("http://files.grouplens.org/datasets/movielens/ml-1m.zip")
            zf = zipfile.ZipFile(dataset_path)
            ratings = pd.read_csv(
                zf.open("ml-1m/ratings.dat"), 
                delimiter = "::", #seperator
                names = ["userId", "movieId", "rating", "timestamp"], 
                engine = "python" #remove ParserWarning: "c" engine not supported
            )
        elif self.type == "100k":
            dataset_path = super().download_data("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
            zf = zipfile.ZipFile(dataset_path)
            ratings = pd.read_csv(
                zf.open("ml-100k/u.data"), 
                delimiter = "\t", #seperator
                names = ["userId", "movieId", "rating", "timestamp"], 
                engine = "python" #remove ParserWarning: "c" engine not supported
            )
        # latest dataset
        elif self.type == "25m":
            dataset_path = super().download_data("http://files.grouplens.org/datasets/movielens/ml-25m.zip")
            zf = zipfile.ZipFile(dataset_path)
            ratings = pd.read_csv(zf.open("ml-25m/ratings.csv"))
        elif self.type == "latest-small":
            dataset_path = super().download_data("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
            zf = zipfile.ZipFile(dataset_path)
            print(zf.__dict__)
            ratings = pd.read_csv(zf.open("ml-latest-small/ratings.csv"))
        elif self.type == "latest":
            dataset_path = super().download_data("http://files.grouplens.org/datasets/movielens/ml-latest.zip")
            zf = zipfile.ZipFile(dataset_path)
            ratings = pd.read_csv(zf.open("ml-latest/ratings.csv"))

        return ratings
        
class EpinionsDataset(Dataset):
    def import_data(self):
        """
        Import Epinions Dataset
        1. Get dataset path
        2. Extract dataset from archive file
        3. Preprocess dataset
        4. Dataset to dataframe
        """
        # Get dataset
        dataset_path = super().download_data("http://deepyeti.ucsd.edu/jmcauley/datasets/epinions/epinions_data.tar.gz")
        tf = tarfile.open(dataset_path , "r:gz") 

        # Preprocess json string so that it is json compatible
        data = tf.extractfile("epinions_data/epinions.json").read()
        data = data.decode("UTF-8")
        data = "[" + data + "]"
        data = data.replace("\n","")
        data = data.replace("}{", "},{")

        # Json string to json
        data = json.loads(json.dumps(eval(data)))

        ratings = pd.json_normalize(data)
        return ratings

class BookCrossingDataset(Dataset):
    def import_data(self):
        """
        Import Book Crossing Dataset
        1. Get dataset path
        2. Extract dataset from archive file
        3. Dataset to dataframe
        """
        # Get dataset
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)
        ratings = pd.read_csv(
            zf.open("BX-Book-Ratings.csv"),
            delimiter = ";",
            encoding = "ISO-8859-1" # Prevent error 'utf-8' codec can't decode byte 0xba in position 183549: invalid start byte
        )

        return ratings


    