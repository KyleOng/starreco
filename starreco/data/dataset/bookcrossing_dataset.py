import zipfile
import requests
import time
import warnings
import logging
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from .dataset import BaseDataset, User, Item, Rating

# Done
# Future work: solve http.client.RemoteDisconnected: Remote end closed connection without response
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
        users = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        user = User(users, "User-ID")
        user.cat_columns = ["Location"]
        user.num_columns = ["Age"]

        # Item dataframe
        books = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        # Get books descriptions
        if self.crawl_data:
            books = self.get_books_with_descriptions(books)
        item = Item(books, "ISBN")
        item.cat_columns = ["Book-Author", "Publisher"]
        if self.crawl_data:
            item.doc_columns = ["description"]

        # Rating dataframe
        ratings = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        rating = Rating(ratings, "Book-Rating", user, item)

        return rating, user, item

    def get_books_with_descriptions(self, books = None):
        books_path = "BX-Books.csv"
        last_valid_index = 0
        
        # Create a logging file using current timestamp.
        timestamp = int(time.time())
        logging.basicConfig(filename = f"bx_logging_{timestamp}.log",
                            format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt = '%Y-%m-%d:%H:%M:%S',
                            level = logging.INFO)
        print(f"Log file created in your working directory at {timestamp}")

        # If path exist, resume from last description valid index
        if os.path.isfile(books_path):
            books = pd.read_csv(books_path, low_memory = False)
            last_valid_index = books["description"].last_valid_index() + 1
            
            print(f"{books_path} exist. Continue from last description value index {last_valid_index}.")
            logging.info(f"{books_path} exist. Continue from last description value index {last_valid_index}.")
        else:
            assert books is not None

        # Start crawling
        for i in tqdm(range(last_valid_index, len(books)), total = len(books)):
            isbn = books.loc[i, "ISBN"]
            description = np.nan
            google_api_link = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn.zfill(10)}"

            attempt = 0
            while attempt < 10: # maximum 10 attempts
                response = requests.get(google_api_link)

                if response.status_code == 200:
                    try:
                        description = response.json()["items"][0]["volumeInfo"]["description"]
                        logging.info(f"Success at ISBN {isbn}.")
                    except Exception as e:
                        logging.warning(f"Success with empty description at ISBN {isbn}")
                    break
                else:
                    attempt += 1
                    warnings.warn(f"Reattempt crawling at ISBN {isbn} for the {attempt}th time(s). Status code {response.status_code}")
                    logging.warning(f"Reattempt crawling at ISBN {isbn} for the {attempt}th time(s). Status code {response.status_code}")
                    time.sleep(5) 

                    if attempt == 10:
                        warnings.warn(f"Fail at ISBN {isbn}. Please check your logging file.")
                        logging.error(f"Fail at ISBN {isbn}.")

            # Insert description and export dataframe
            books.loc[i, "description"] = description
            books.to_csv(books_path, index = False)

        print("End crawling")
        return books
