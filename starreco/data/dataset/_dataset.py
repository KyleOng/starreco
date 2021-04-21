import urllib
import os
import re

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

class BaseDataset:
    """
    Base class for dataset.
    """
    
    _download_path = os.path.realpath(__file__).replace("data/dataset/_dataset.py", "dataset/")

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


