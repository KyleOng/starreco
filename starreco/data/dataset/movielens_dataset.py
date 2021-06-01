import zipfile
import requests
import time
import warnings
import os
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

from .dataset import BaseDataset, User, Item, Rating


class MovielensDataset(BaseDataset):
    """
    Dataset class for Movielens.
    """

    def import_data(self):
        """
        Import Movielens rating, user and movie dataframes. 
        """
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        users = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::", names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        user = User(users, "userId")
        user.cat_columns = ["gender", "occupation", "age"]

        # Item dataframe
        movies = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::", names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        # Crawl movies data
        movies = self.crawl_movies_data(movies)
        movies["genre"] = movies["genre"].apply(lambda x:set(x.split("|")))
        item = Item(movies, "movieId")
        item.set_columns = ["genre"]

        # Rating dataframe
        rating = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
                             names = ["userId", "movieId", "rating", "timestamp"], engine = "python")
        rating = Rating(rating, "rating", user, item)

        return rating, user, item

    def crawl_movies_data(self, movies_1m):
        """
        Crawling movie data.
        """
        movies_path = self._download_path+"ml-1m-movies.csv"
        movielinks_extra_path = self._download_path+"ml-1m_links.csv"

        if not os.path.isfile(movielinks_extra_path):
            raise Exception("""
            Please manually download 'ml-1m_links.csv' from the link here https://drive.google.com/file/d/1k0GTgery8Pyjo3z_igWyQsJ4XeiCBab7/view?usp=sharing and place in 'starreco/dataset' directory.
            """)

        if os.path.isfile(movies_path):            
            movies = pd.read_csv(movies_path, encoding = "ISO-8859-1", engine = "python")
        else:
            # Merge movielen-25m movies.csv and links.csv
            ml_25m_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-25m.zip")
            ml_25m = zipfile.ZipFile(ml_25m_path)
            movies_25m = pd.read_csv(ml_25m.open(f"ml-25m/movies.csv"), encoding = "ISO-8859-1", engine = "python")
            links_25m = pd.read_csv(ml_25m.open(f"ml-25m/links.csv"), encoding = "ISO-8859-1", engine = "python")
            movielinks_25m = movies_25m.merge(links_25m, on = "movieId")

            # Uncomment the code below for testing.
            # Use the export csv from the code below to check between alternative titles.
            """
            import difflib
            movielinks_25m = movielinks_25m[["movieId", "title", "imdbId"]]
            movies = movies_1m.merge(movielinks_25m, on = ["movieId"], how = "left")
            temp = movies[movies["title_x"] != movies["title_y"]][~movies["imdbId"].isna()][["title_x","title_y"]]
            # Using python difflib library to find the similarity between titles
            temp["similarity"] = temp.apply(lambda row: (lambda seq = 
                                    difflib.SequenceMatcher(None, row["title_x"], row["title_y"]):
                                    "%.2f" % seq.quick_ratio())(),
                                    axis = 1)
            temp = temp.sort_values("similarity")
            temp.to_csv("movies_25m_1m_title_check.csv", index = False)
            """

            # Uncomment the code below for testing.
            # Use the export csv from the code below to manually insert imdbId.
            """
            temp = movies[movies["imdbId"].isna()]
            temp.to_csv("movies_25m_1m_empty_links.csv", index = False)
            """

            # Merge movielens-1m movies and movielens-25m movielinks
            movielinks_25m = movielinks_25m[["movieId", "imdbId"]]
            movielinks_extra = pd.read_csv(movielinks_extra_path, encoding = "ISO-8859-1", engine = "python")
            movies = movies_1m.merge(movielinks_25m, on = "movieId", how = "left")
            movies.update(movies[["movieId"]].merge(movielinks_extra, on = ["movieId"], how = "left"))
            # Create new column on movies
            movies["plot"] = np.nan

            # Create a logging file using timestamp.
            timestamp = int(time.time())
            logging.basicConfig(filename = f"movies_logging_{timestamp}.log",
                                format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                datefmt = '%Y-%m-%d:%H:%M:%S',
                                level = logging.INFO)
            print(f"Log file created in your working directory at {timestamp}")

            # Start crawling
            print("Start crawling")
            for i, movie in tqdm(enumerate(movies.itertuples()), total = len(movies["imdbId"])):
                movie_id = movie[1]
                imdb_id = movie[4]
                plot = ""
                if not pd.isna(imdb_id):
                    attempt = 0
                    while attempt < 10: # maximum 10 attempts
                        try:
                            # Plot crawling from IMDB site
                            imdb_link = f"https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}"
                            response = requests.get(imdb_link)
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Uncomment the code below for testing.
                            # Use the export txt to find plot throught html
                            """
                            with open(f"soup_{movie_id}.txt", "w") as text_file:
                                text_file.write(soup.prettify())
                            """

                            # Find plot from soup
                            plot = soup.find_all("div", class_="GenresAndPlot__TextContainerBreakpointXS-cum89p-0")[0].text  

                            if plot: 
                                logging.info(f"Success at movieId {movie_id}.")
                            else:
                                logging.warning(f"Success with empty plot at movieId {movie_id}")

                            break   
                        except Exception as e:
                            attempt += 1
                            warnings.warn(f"Reattempt crawling at movieId {movie_id} for the {attempt}th time(s). {e}")
                            logging.warning(f"Reattempt crawling at movieId {movie_id} for the {attempt}th time(s). {e}")
                            time.sleep(5) 

                            if attempt == 10:
                                warnings.warn(f"Fail at movieId {movie_id}. Please check your logging file.")
                                logging.error(f"Fail at movieId {movie_id}.")
                else:
                    # If imdbId is empty, proceed to the next movieId and provide a warning log.    
                    logging.warning(f"Empty imdbId at movieId {movie_id}")   

                # Insert plot and export dataframe
                movies.loc[i, "plot"] = plot
                movies.to_csv(movies_path, index = False)
            print("End crawling")
        return movies           

