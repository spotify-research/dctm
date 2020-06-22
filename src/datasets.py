#
# Copyright 2020 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Datasets utilities.

If you use nltk you may need the following:
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')
"""
import os

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.utils import Bunch

ENGLISH_WORDS = set(nltk.corpus.words.words())
STEMMER = SnowballStemmer('english')


class LemmaTokenizer:
    def __init__(self, stem=False):
        self.wnl = WordNetLemmatizer()
        if stem:
            self.stemmer = SnowballStemmer('english')
        else:
            self.stemmer = Bunch(stem=lambda x: x)

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(self.stemmer.stem(t))
            for t in word_tokenize(doc) if t.lower() in ENGLISH_WORDS
        ]


def get_neurips(filename: str):
    """Get NeurIPS dataset.

    Args:
        filename (str): Location of the file for NeurIPS dataset.
    """
    df = pd.read_csv(filename, header=0, index_col=0)
    year = np.array([x.split('_')[0] for x in df.columns])

    # preprocess
    df = df.loc[df.index.dropna()]
    df = df.loc[~df.index.isin(ENGLISH_STOP_WORDS)]
    df.index = [STEMMER.stem(x) for x in df.index.tolist()]

    # merge same words together
    df = df.groupby(level=0).sum()

    vocabulary = df.sum(axis=1)
    return df, year, vocabulary


def get_sotu(path: str, stem=False):
    df = {}
    for filename in sorted(os.listdir(path)):
        fn = os.path.join(path, filename)
        df[filename] = ' '.join(
            [x.decode("utf-8") for x in open(fn, 'rb').readlines()])

    df = pd.Series(df)
    df.index = df.index.str.split('.txt').map(lambda x: x[0])
    df = pd.DataFrame(df, columns=['text'])
    df['years'] = df.index.str.split('_').map(lambda x: int(x[1]))
    df['author'] = df.index.str.split('_').map(lambda x: x[0])
    stopwords_english = LemmaTokenizer(stem=stem)(
        ' '.join(list(ENGLISH_STOP_WORDS)))

    vect = CountVectorizer(
        max_df=0.9, min_df=50, stop_words=stopwords_english,
        tokenizer=LemmaTokenizer(stem=stem))
    corpus = vect.fit_transform(df.text)
    vocabulary = np.array(vect.get_feature_names())

    keep = np.array(corpus.sum(axis=1) > 0).flatten()
    corpus = corpus[keep]
    df = df.loc[keep]
    return df, corpus, vocabulary

import json
def get_doj(filename: str = 'data/doj.json', stem=True, min_counts=50):
    df = []
    with open(filename, 'r') as f:
        for line in f:
            df.append(json.loads(line))

    df = pd.DataFrame(df).set_index('id')
    df.index = range(df.shape[0])

    df['text'] = df.title + ' ' + df.contents
    days = pd.to_datetime(
        df.date.str.split('T').map(lambda x: x[0]).str.split('-').map(
            lambda x: '-'.join(x[:-1])), format='%Y-%m')
    df['days'] = days
    df['time_delta'] = (df.days - df.days.min()).dt.days

    stop_words = LemmaTokenizer(stem=stem)(
        ' '.join(list(ENGLISH_STOP_WORDS)))

    vectorizer = CountVectorizer(
        max_df=0.85, min_df=min_counts, stop_words=stop_words,
        tokenizer=LemmaTokenizer(stem=stem))
    corpus = vectorizer.fit_transform(df.text)
    vocabulary = np.array(vectorizer.get_feature_names())

    keep = np.array(corpus.sum(axis=1) > 0).flatten()
    corpus = corpus[keep]
    df = df.loc[keep]
    return df, corpus, vocabulary


def train_test_split(X, index_points, train_size=0.75, return_sorted=True):
    unique_index_points = np.unique(index_points)
    train_idx = np.random.choice(
        unique_index_points, int(len(unique_index_points) * train_size),
        replace=False)
    tr_idx = np.array([x in train_idx for x in index_points.flatten()])
    index_tr = index_points[tr_idx]
    X_tr = X[tr_idx]

    test_idx = np.unique(list(set(unique_index_points) - set(train_idx)))
    ts_idx = np.array([x in test_idx for x in index_points.flatten()])
    index_ts = index_points[ts_idx]
    X_ts = X[ts_idx]

    idx = np.argsort(index_tr, axis=0).flatten()
    X_tr_sorted = X_tr[idx]
    index_tr_sorted = index_tr[idx]

    idx = np.argsort(index_ts, axis=0).flatten()
    X_ts_sorted = X_ts[idx]
    index_ts_sorted = index_ts[idx]

    return_list = [X_tr, X_ts, index_tr, index_ts]
    if return_sorted:
        return_list += [
            X_tr_sorted, X_ts_sorted, index_tr_sorted, index_ts_sorted
        ]

    return return_list


def print_to_file_for_gdtm(df, vocabulary, corpus, filename='test', path='.'):
    """Utility function to save datasets for gDTM.

    Args:
        df ([type]): [description]
        vocabulary ([type]): [description]
        corpus ([type]): [description]
        filename (str, optional): [description]. Defaults to 'test'.
    """
    with open(os.path.join(path, '{}_corpus.txt'.format(filename)), 'w') as f:
        n_times = df.years.unique().size
        f.writelines('{}\n'.format(n_times))
        for name, group in df.groupby('years')[0]:
            n_docs = group.shape[0]
            f.writelines('{}\n{}\n'.format(name.timestamp(), n_docs))

            idx = group.index.values
            # np.array([df.index.get_loc(x) for x in group.index])
            for c in corpus[idx]:
                d = c.todok()
                f.writelines(
                    str(len(d)) + ' ' + ' '.join(
                        '{}:{}'.format(x[1], int(v))
                        for x, v in d.items()) + '\n')

    with open(os.path.join(path, '{}_lexicon.txt'.format(filename)), 'w') as f:
        f.writelines('\n'.join(vocabulary))
