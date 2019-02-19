# coding: utf-8
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import itertools
import pandas as pd
from networkx.readwrite import json_graph
from scipy import spatial
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:
    def __init__(self,
                 source_file_path,
                 sep='\t',
                 language='english',
                 n_gram=1,
                 vectorization='tfidf',
                 max_relative_frequency=1.,
                 min_absolute_frequency=0,
                 max_features=2000,
                 sample=None,
                 text_col=None,
                 full_text_col=None,
                 title_col=None,
                 author_col=None,
                 affiliation_col=None,
                 date_col=None,
                 ):

        if isinstance(source_file_path, str) or isinstance(source_file_path, Path):
            self._source_file_path = source_file_path
            self.data_frame = pd.read_csv(source_file_path, sep=sep, encoding='utf-8')
        elif isinstance(source_file_path, pd.DataFrame):
            self._source_file_path = 'pd.DataFrame from memory'
            self.data_frame = source_file_path.copy()

        self._sep = sep
        self._language = language
        self._n_gram = n_gram
        self._vectorization = vectorization
        self._max_relative_frequency = max_relative_frequency
        self._min_absolute_frequency = min_absolute_frequency
        self._sample = sample

        self._text_col = text_col or 'text'
        self._full_text_col = full_text_col or self._text_col
        self._title_col = title_col or 'title'
        self._author_col = author_col or 'author'
        self._affiliation_col = affiliation_col or 'affiliation'
        self._date_col = date_col or 'date'

        self.max_features = max_features
        if sample:
            if isinstance(sample, bool):
                sample = 0.8
            if isinstance(sample, float):
                self.data_frame = self.data_frame.sample(frac=sample)
            else:
                raise ValueError(f'Unknown sample: {sample}')
        # reset index because row numbers are used to access rows
        self.data_frame = self.data_frame.reset_index()
        for col in ['index', 'id', 'docnum']:
            # remove these columns because they are not needed
            if col in self.data_frame.columns:
                self.data_frame = self.data_frame.drop(col, axis=1)
        # fill in null values
        self.data_frame = self.data_frame.fillna(' ')
        # get shape of df (previous code used count, which won't work if there are columns other than index 0 that have nans)
        self.size = self.data_frame.shape[0]

        stop_words = []
        if language is not None:
            stop_words = stopwords.words(language)
        if vectorization == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        elif vectorization == 'tf':
            vectorizer = CountVectorizer(ngram_range=(1, n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        else:
            raise ValueError(f'Unknown vectorization type: {vectorization}')
        self.vectorizer = vectorizer
        self.sklearn_vector_space = vectorizer.fit_transform(self.data_frame[self._text_col].tolist())
        self.gensim_vector_space = None
        vocab = vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def export(self, file_path):
        self.data_frame.to_csv(path_or_buf=file_path, sep=self._sep, encoding='utf-8')

    def full_text(self, doc_id):
        return self.data_frame.iloc[doc_id][self._full_text_col]

    def title(self, doc_id):
        return self.data_frame.iloc[doc_id][self._title_col]

    def date(self, doc_id):
        return self.data_frame.iloc[doc_id][self._date_col]

    def author(self, doc_id):
        aut_str = str(self.data_frame.iloc[doc_id][self._author_col])
        return aut_str.split(', ')

    def affiliation(self, doc_id):
        aff_str = str(self.data_frame.iloc[doc_id][self._affiliation_col])
        return aff_str.split(', ')

    def documents_by_author(self, author, date=None):
        ids = []
        potential_ids = range(self.size)
        if date:
            potential_ids = self.doc_ids(date)
        for i in potential_ids:
            if self.is_author(author, i):
                ids.append(i)
        return ids

    def all_authors(self):
        author_list = []
        for doc_id in range(self.size):
            author_list.extend(self.author(doc_id))
        return list(set(author_list))

    def is_author(self, author, doc_id):
        return author in self.author(doc_id)

    def docs_for_word(self, word_id):
        ids = []
        for i in range(self.size):
            vector = self.vector_for_document(i)
            if vector[word_id] > 0:
                ids.append(i)
        return ids

    def doc_ids(self, date):
        return self.data_frame[self.data_frame[self._date_col] == date].index.tolist()

    def vector_for_document(self, doc_id):
        vector = self.sklearn_vector_space[doc_id]
        cx = vector.tocoo()
        weights = [0.0] * len(self.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weights[word_id] = weight
        return weights

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def id_for_word(self, word):
        for i, s in self.vocabulary.items():
            if s == word:
                return i
        return -1

    def similar_documents(self, doc_id, num_docs):
        doc_weights = self.vector_for_document(doc_id)
        similarities = []
        for a_doc_id in range(self.size):
            if a_doc_id != doc_id:
                similarity = 1.0 - spatial.distance.cosine(doc_weights, self.vector_for_document(a_doc_id))
                similarities.append((a_doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]

    def find_similar_document_pairs(self, threshold):
        '''find all the similar document pairs above a threshold.
        for a tf vectorized corpus, try threshold=0.83
        for a tfidf vectorized corpus, try threshold=0.88
        '''
        # compute pairwise similarities
        result = np.dot(normalize(self.sklearn_vector_space, axis=1),
                        normalize(self.sklearn_vector_space, axis=1).T)
        # keep similarities above threshold
        similar_pairs = np.where(result.toarray() > threshold)
        # remove self pairs
        similar_pairs = [frozenset([v0, v1]) for v0, v1 in zip(similar_pairs[0], similar_pairs[1]) if v0 != v1]
        # remove duplicate pairs
        similar_pairs = list(set(similar_pairs))
        # return as a list of tuples
        similar_pairs = [(list(i)[0], list(i)[1]) for i in similar_pairs]
        return similar_pairs

    # def collaboration_network(self, doc_ids=None, nx_format=False):
    #     nx_graph = nx.Graph(name='')
    #     for doc_id in doc_ids:
    #         authors = self.author(doc_id)
    #         for author in authors:
    #             nx_graph.add_node(author)
    #         for i in range(0, len(authors)):
    #             for j in range(i + 1, len(authors)):
    #                 nx_graph.add_edge(authors[i], authors[j])
    #     bb = nx.betweenness_centrality(nx_graph)
    #     nx.set_node_attributes(nx_graph, bb, 'betweenness')
    #     if nx_format:
    #         return nx_graph
    #     else:
    #         return json_graph.node_link_data(nx_graph)
