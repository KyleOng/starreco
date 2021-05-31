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
        movies_path = self._download_path+"ml-1m-movies.csv"

        if os.path.isfile(movies_path):            
            movies = pd.read_csv(movies_path, encoding = "ISO-8859-1", engine = "python")
        else:
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

            movielinks_25m = movielinks_25m[["movieId", "imdbId"]]
            movielinks_extra = pd.read_csv(self._download_path+"ml-1m_links.csv", encoding = "ISO-8859-1", engine = "python")
            movies = movies_1m.merge(movielinks_25m, on = "movieId", how = "left")
            movies.update(movies[["movieId"]].merge(movielinks_extra, on = ["movieId"], how = "left"))
            movies["plot"] = np.nan

            print("Log file created in your working directory.")
            logging.basicConfig(filename="movies.log", level=logging.INFO)

            for i, movie in tqdm(enumerate(movies.itertuples()), total = len(movies["imdbId"])):
                movie_id = movie[1]
                imdb_id = movie[4]
                if not pd.isna(imdb_id):
                    trial = 0
                    while trial < 4: # maximum 10 trials
                        try:
                            imdb_link = f"https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}"
                            response = requests.get(imdb_link)
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Uncomment the code below for testing.
                            # Use the export txt to find plot throught html
                            """
                            with open("output.txt", "w") as text_file:
                                text_file.write(soup.prettify())
                            """
                            if i == 2:
                                raise Exception (testing)

                            plot = soup.find_all("div", class_="GenresAndPlot__TextContainerBreakpointXS-cum89p-0")[0].text
                            break
                        except Exception as e:
                            trial += 1
                            warnings.warn(f"Fail at movieId {movie_id}. Reattempt crawling for the {trial}th time(s). {e}")
                            logging.error(f"Fail at movieId {movie_id}. Reattempt crawling for the {trial}th time(s). {e}")
                            time.sleep(10)
                else:
                    plot = ""
                movies.loc[i, "plot"] = plot
                logging.info(f"Success at movieId {movie_id}.")

            movies.to_csv(movies_path, index = False)
        import pdb 
        pdb.set_trace()
        return movies           

