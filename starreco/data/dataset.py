import urllib
import os
import json
import re
import zipfile, tarfile

from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl 
import requests

class Dataset:
    datasets_path = "starreco/dataset/"
    
    def __init__(self):
        pass

    def download_data(self, url):
        file_name = url.split("/")[-1]
        dataset_path = f"{self.datasets_path}{file_name}"
        if not os.path.isfile(dataset_path):
            try:
                # Streaming, so we can iterate over the response.
                response = requests.get(url, stream=True)
                total_size_in_bytes= int(response.headers.get("content-length", 0))
                block_size = 1024 #1 Kilobyte
                progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

                with open(dataset_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()

                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    raise Exception("Something went wrong during data download.")
            except KeyboardInterrupt:
                if os.path.isfile(dataset_path):
                    os.remove(dataset_path)
                raise KeyboardInterrupt("Not fully downloaded dataset has been deleted. Please redownload the dataset.")
        return dataset_path

class MovielensDataset(Dataset):
    def __init__(self, type = "1m"):
        super().__init__()
        self.type = type
            
    def import_data(self):
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
        dataset_path = super().download_data("http://deepyeti.ucsd.edu/jmcauley/datasets/epinions/epinions_data.tar.gz")
        tf = tarfile.open(dataset_path , "r:gz") 

        data = tf.extractfile("epinions_data/epinions.json").read()
        data = data.decode("UTF-8")
        data = "[" + data + "]"
        data = data.replace("\n","")
        data = data.replace("}{", "},{")

        data = json.loads(json.dumps(eval(data)))
        ratings = pd.json_normalize(data)
        return ratings

class BookCrossingDataset(Dataset):
    def import_data(self):
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)
        ratings = pd.read_csv(
            zf.open("BX-Book-Ratings.csv"),
            delimiter = ";",
            encoding = "ISO-8859-1" # Prevent error 'utf-8' codec can't decode byte 0xba in position 183549: invalid start byte
        )

        return ratings


    