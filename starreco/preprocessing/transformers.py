import os
import operator

import numpy as np
import nltk
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from scipy.sparse import lil_matrix
from tqdm import tqdm

# Done
class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    """
    Custom MultiLabelBinarizer.

    Notes: Original MultiLabelBinarizer fit_transform() only takes 2 positional arguments. However, our custom pipeline assumes the MultiLabelBinarizer fit_transform() is defined to take 3 positional arguments. Hence, adding an additional argument y to fit_transform() fix the problem.
    """

    def fit(self, X, y = None):
        return super().fit(X.flatten())

    def transform(self, X, y = None):
        return super().transform(X.flatten())

    def fit_transform(self, X, y = None):
        return super().fit_transform(X.flatten())

# Testing
class SetTransformer(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return self.feature_names

    def fit(self, X, y = None):
        # Reset column transformer for every fit
        self.column_transformer = ColumnTransformer([])

        for column in X.columns: 
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
                ("binarizer", CustomMultiLabelBinarizer(sparse_output = True))
            ])
            self.column_transformer.transformers.append(
                (column, pipe, [column])
            )

        self.column_transformer.fit(X)

        self.feature_names = []
        for column in X.columns: 
            self.feature_names = np.concatenate((self.feature_names,
                                                self.column_transformer.named_transformers_[column]\
                                                .named_steps["binarizer"].classes_),
                                                axis = None)

        return self

    def transform(self, X, y = None):
        return self.column_transformer.transform(X)

# Done
class DocTransformer(BaseEstimator, TransformerMixin):
    """
    Document Transformer.
    """

    word_embeddings = {}

    def __init__(self, 
                 max_length:int = 300, 
                 max_df:float = 0.5, 
                 vocab_size:int = 8000,
                 glove_path:str = "glove.6B/glove.6B.50d.txt"):
        assert max_length > 0 and max_df > 0 and vocab_size > 0

        self.max_length = max_length
        self.max_df = max_df
        self.vocab_size = vocab_size
        self.glove_path = glove_path

    def _get_feature_names(self, X):
        words = []
        for column in X.columns:
            words = np.concatenate((words,
                                   self.column_transformer\
                                   .named_transformers_[column]\
                                   .named_steps["vectorizer"].get_feature_names()),
                                   axis = None)
        return words

    def get_feature_names(self):
        return self.feature_names

    def fit(self, X, y = None):
        # Reset column transformer for every fit
        self.column_transformer = ColumnTransformer([])

        for column in X.columns:
            vectorizer =  TfidfVectorizer(analyzer = "word", 
                                          tokenizer = nltk.word_tokenize,
                                          max_features = None, 
                                          stop_words = "english") 
            pipe = Pipeline([
                ("imputer", FunctionTransformer(lambda row:row.fillna("").str[:self.max_length])),
                ("vectorizer", vectorizer)
            ])         
            self.column_transformer.transformers.append(            
                (column, pipe, column)
            )
        self.column_transformer.fit(X)

        # Calculate TFIDF
        tfidf = self.column_transformer.transform(X)
        words = self._get_feature_names(X)
        weights = np.ravel(tfidf.mean(axis = 0))
        word_weights = {word: weight for word, weight in zip(words, weights)}

        # Remove corpus specific stop words that have the TFIDF document frequency higher than 0.5 (default).
        non_stop_word_weights = {word: weight for word, weight in word_weights.items() if weight < self.max_df}

        # Keep words in vocabulary.
        # Load pretrained word embeddings model.
        if not bool(self.word_embeddings):
            num_lines = sum(1 for _ in open(self.glove_path, 'r', encoding="utf-8"))
            with open(self.glove_path, 'r', encoding="utf-8") as f:
                f_tqdm = tqdm(f, total = num_lines)
                f_tqdm.set_description("Loading glove.6B/glove.6B.50d.txt")
                for line in f_tqdm:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    self.word_embeddings[word] = vector
        vocabs = list(self.word_embeddings.keys())  
        vocab_word_weights = {word: weight for word, weight in non_stop_word_weights.items() if word in vocabs}

        # Top 8000 (default) distinct words.
        sort_word_weights = dict(sorted(vocab_word_weights.items(), key= operator.itemgetter(1), reverse = True))
        top_word_weights = {word: vocab_word_weights[word] for word in list(sort_word_weights.keys())[:self.vocab_size]}

        
        self.feature_names = top_words = list(top_word_weights.keys())
        return self

    def transform(self, X, y = None):
        tfidf = self.column_transformer.transform(X)

        words = self._get_feature_names(X)
        top_words_tqdm = tqdm(enumerate(self.feature_names), total = self.vocab_size)
        top_tfidf = lil_matrix((tfidf.shape[0], len(self.feature_names)), dtype = tfidf.dtype)
        
        for i, top_word in top_words_tqdm:
            j = np.where(words == top_word)[0][0]
            top_tfidf[:, i] = tfidf[:, j]
            top_words_tqdm.set_description(f"Indexing {top_word}: {j} -> {i}")

        return top_tfidf != 0