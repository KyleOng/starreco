import os
import operator

import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Add current working directory to nltk path
nltk.data.path.append(os.getcwd())

# Done
class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    """
    Custom MultiLabelBinarizer.

    Notes: Original MultiLabelBinarizer fit_transform() only takes 2 positional arguments. However, our custom pipeline assumes the MultiLabelBinarizer fit_transform() is defined to take 3 positional arguments. Hence, adding an additional argument y to fit_transform() fix the problem.
    """

    def fit(self, X, y = None):
        return super().fit(np.ravel(X))

    def transform(self, X, y = None):
        return super().transform(np.ravel(X))

    def fit_transform(self, X, y = None):
        return super().fit_transform(np.ravel(X))

# Testing
class CustomMultiColumnLabelEncoder(LabelEncoder):
    """
    Custom MultiColumnLabelEncoder

    Notes: Original LabelEncoder only accept 1-d array. Hence, by looping through the input X and apply LabelEncoder on each fix the problem.
    """
    
    def fit(self, X, y = None):
        return self # not relevant

    def transform(self, X):
        # reset array for each transform
        self.label_encoders = []

        output = X.copy()
        for i in range(X.shape[-1]):
            label_encoder = LabelEncoder()
            output[:,i] = label_encoder.fit_transform(X[:,i])
            self.label_encoders.append(label_encoder)
        return output

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

# Done
class SetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class for transforming set type data.
    """

    def get_feature_names(self):
        return self.feature_names 

    def fit(self, X, y = None):
        # Reset column transformer for every fit
        self.column_transformer = ColumnTransformer([])
        for column in X.columns: 
            pipe = Pipeline([("imputer", SimpleImputer(strategy = "constant", fill_value = {})),
                             ("binarizer", CustomMultiLabelBinarizer(sparse_output = True))])
            self.column_transformer.transformers.append((column, pipe, [column]))
        
        # Fit
        self.column_transformer.fit(X)

        # Get feature names based on fitted X
        self.feature_names = []
        for column in X.columns: 
            new_feature_names = self.column_transformer.named_transformers_[column].named_steps["binarizer"].classes_
            new_feature_names = [ f"{column}_{new_feature_name}" for new_feature_name in new_feature_names]
            self.feature_names = np.concatenate((self.feature_names, new_feature_names), axis = None)

        return self

    def transform(self, X, y = None):
        # Transform
        return self.column_transformer.transform(X)

# Done
class DocTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class for transforming document type data.

    max_char (int): Maximum length of document characters of each item. Default: 300.
    max_df (float): Terms will be ignored that have a document frequency higher than the given threshold. Default: 0.5.
    vocab_size (int): Vocabulary size. Default: 8000.
    """

    max_lens_ = {}
    
    def __init__(self, 
                 max_char:int = 300, 
                 max_df:float = 0.5, 
                 vocab_size:int = 8000,
                 glove_path:str = "glove.6B/glove.6B.50d.txt",
                 progress_bar = True):
        assert max_char > 0 and max_df > 0 and vocab_size > 0

        self.max_char = max_char
        self.max_df = max_df
        self.vocab_size = vocab_size
        self.glove_path = glove_path
        self.progress_bar = progress_bar

    def fit(self, X, y = None):
        # Build and reset column transformer for every fit
        self.column_transformer = ColumnTransformer([])
        for column in X.columns:
            vectorizer =  TfidfVectorizer(analyzer = "word", 
                                          tokenizer = nltk.word_tokenize,
                                          max_features = None, 
                                          stop_words = "english",) 
            pipe = Pipeline([("imputer", FunctionTransformer(lambda column:column.fillna("").str[:self.max_char])),
                             ("vectorizer", vectorizer)])         
            self.column_transformer.transformers.append((column, pipe, column))

        # Calculate TFIDF
        if self.progress_bar: 
            print("[Fit] Transforming TFIDF".ljust(38) + ": ...", end='\r')
        tfidf = self.column_transformer.fit_transform(X)

        # Get vocabs from X input
        vocabs_ = []
        for column in X.columns:
            vocabs_ = np.concatenate((vocabs_,
                                      self.column_transformer\
                                      .named_transformers_[column]\
                                      .named_steps["vectorizer"].get_feature_names()),
                                     axis = None)

        # Get associate TFIDF frequency/term/weight for each vocab
        weights = np.ravel(tfidf.mean(axis = 0))
        vocab_weights = {word: weight for word, weight in zip(vocabs_, weights)}
        if self.progress_bar: 
            print("[Fit] Transforming TFIDF".ljust(38) + ": 100%")

        # Remove corpus specific stop words that have the TFIDF document frequency higher than 0.5 (default).
        if self.progress_bar: 
            vocab_weights_tqdm = tqdm(vocab_weights.items(), total = len(vocab_weights.items()), bar_format = "{desc:}{percentage:3.0f}%|{bar:10}{r_bar}")
            vocab_weights_tqdm.set_description("[Fit] Removing stopwords".ljust(38))
        else:
            vocab_weights_tqdm = vocab_weights.items()
        vocab_weights = {word: weight for word, weight in vocab_weights_tqdm if weight < self.max_df}

        # Load pretrained word embeddings model.
        vocabs = []
        num_lines = sum(1 for _ in open(self.glove_path, 'r', encoding="utf-8"))
        with open(self.glove_path, 'r', encoding="utf-8") as f:
            if self.progress_bar: 
                f_tqdm = tqdm(f, total = num_lines, bar_format = "{desc:}{percentage:3.0f}%|{bar:10}{r_bar}")
                f_tqdm.set_description("[Fit] Loading pretrained glove weights".ljust(38))
            else:
                f_tqdm = f
            for line in f_tqdm:
                values = line.split()
                word = values[0]
                vocabs.append(word)
                
        # Keep vocabs which existed in pretrained word embedding vocabularies.
        if self.progress_bar: 
            vocab_weights_tqdm = tqdm(vocab_weights.items(), total = len(vocab_weights.items()), bar_format = "{desc:}{percentage:3.0f}%|{bar:10}{r_bar}")
            vocab_weights_tqdm.set_description("[Fit] Keeping words in vocabularies".ljust(38))
        else:
            vocab_weights_tqdm = vocab_weights.items()
        vocab_weights = {vocab: weight for vocab, weight in vocab_weights_tqdm if vocab in vocabs}

        # Top 8000 (default) distinct words.
        vocab_weights = dict(sorted(vocab_weights.items(), key = operator.itemgetter(1), reverse = True))
        if self.progress_bar: 
            vocabs_tqdm = tqdm(list(vocab_weights.keys())[:self.vocab_size], total = self.vocab_size, bar_format = "{desc:}{percentage:3.0f}%|{bar:10}{r_bar}")
            vocabs_tqdm.set_description("[Fit] Keeping 8000 distinct words".ljust(38))
        else:
            vocabs_tqdm = list(vocab_weights.keys())[:self.vocab_size]
        vocab_weights = {vocab: vocab_weights[vocab] for vocab in vocabs_tqdm}

        # Create vobabulary mapper.
        vocabs = list(vocab_weights.keys())
        self.vocab_map = {vocab: i for i, vocab in enumerate(vocabs, start = 2)}

        return self

    def transform(self, X, y = None):
        def sentences_to_indices(x, vocab_map):
            """
            Transform sentences to indices based on the vocabulary mapper.
            For example, "This document is the first document"-> [1,2,3,4,5,2]

            Note: vocab map index start from 2, as 1s are for unknown vocab and 0s for padding. 
            """
            def sentence_to_indices(sentence):
                tokens = nltk.word_tokenize(str(sentence))
                # Remove stop words
                non_stops = [token for token in tokens if token.lower() not in stopwords.words('english')]
                word_indices = [vocab_map[non_stop.lower()] if non_stop.lower() in vocab_map else 1 for non_stop in non_stops]
                return word_indices

            if self.progress_bar: 
                tqdm.pandas(desc = f"[Transform] Indexing {x.name} sentences".ljust(len(x.name)+38), 
                            bar_format = "{desc:}: {percentage:3.0f}%|{bar:10}{r_bar}")
                x_word_indices = x.progress_apply(lambda sentence: sentence_to_indices(sentence))
            else:
                x_word_indices = x.apply(lambda sentence: sentence_to_indices(sentence))

            return x_word_indices

        def pad_zeros(x):
            """
            Zero padding word indices until it reached the maximum length.
            For example, [[1,2,3],[1,2,3,4],[1,2]] -> [[1,2,3,0],[1,2,3,4],[1,2,0,0]]
            """
            max_len = x.apply(lambda word_indices: len(word_indices)).max()

            # Save transform max len
            self.max_lens_[x.name] = max_len

            if self.progress_bar: 
                tqdm.pandas(desc = f"[Transform] Zero padding {x.name} word indices".ljust(len(x.name)+38), 
                            bar_format = "{desc:}: {percentage:3.0f}%|{bar:10}{r_bar}")
                x_pad = x.progress_apply(lambda word_indices: np.pad(word_indices, (0, max_len - len(word_indices))))
            else:
                x_pad = x.apply(lambda word_indices: np.pad(word_indices, (0, max_len - len(word_indices))))

            return x_pad

        # Vocabulary mapper for sentence indexing.
        X = X.apply(lambda x:sentences_to_indices(x, self.vocab_map))

        # Zero padding
        X = X.apply(lambda x:pad_zeros(x))

        # Transform to sparse for memory efficient
        sparse_stack = csr_matrix(np.vstack(X.values.tolist()))

        return sparse_stack
