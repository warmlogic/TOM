# coding: utf-8
from pathlib import Path
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
# import networkx as nx
# from networkx.readwrite import json_graph


class Corpus:
    def __init__(self,
                 source_filepath,
                 sep: str = '\t',
                 language: str = 'english',
                 n_gram: int = 1,
                 vectorization: str = 'tfidf',
                 max_relative_frequency: float = 1.0,
                 min_absolute_frequency: int = 0,
                 max_features: int = 2000,
                 sample: float = 1.0,
                 text_col: str = None,
                 full_text_col: str = None,
                 title_col: str = None,
                 author_col: str = None,
                 affiliation_col: str = None,
                 dataset_col: str = None,
                 date_col: str = None,
                 id_col: str = None,
                 ):

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
        self._dataset_col = dataset_col or 'dataset'
        self._date_col = date_col or 'date'
        self._id_col = id_col

        self.max_features = max_features

        if isinstance(source_filepath, str) or isinstance(source_filepath, Path):
            self._source_filepath = source_filepath
            self.data_frame = pd.read_csv(
                source_filepath,
                sep=self._sep,
                encoding='utf-8',
                parse_dates=[self._date_col],
            )
        elif isinstance(source_filepath, pd.DataFrame):
            self._source_filepath = 'pd.DataFrame from memory'
            self.data_frame = source_filepath.copy()

        if self._sample < 1.0:
            self.data_frame = self.data_frame.sample(frac=self._sample)

        # reset index because row numbers are used to access rows
        self.data_frame = self.data_frame.reset_index()
        self.data_frame.index.name = 'id'
        for col in ['index', 'id', 'docnum']:
            # remove these columns because they are not needed
            if col in self.data_frame.columns:
                self.data_frame = self.data_frame.drop(col, axis=1)
        # fill in null values
        self.data_frame = self.data_frame.fillna('')
        # get shape of df (previous code used count, which won't work if there are columns other than index 0 that have nans)
        self.size = self.data_frame.shape[0]

        # print(f'Number of unique words: {len(set(np.hstack(self.data_frame[self._text_col].str.split().values))):,}')
        # if self.max_features:
        #     print(f'Reducing vocabulary to: {self.max_features:,}')

        stop_words = []
        if self._language:
            stop_words = stopwords.words(self._language)
        if self._vectorization == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, self._n_gram),
                                         max_df=self._max_relative_frequency,
                                         min_df=self._min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        elif self._vectorization == 'tf':
            vectorizer = CountVectorizer(ngram_range=(1, self._n_gram),
                                         max_df=self._max_relative_frequency,
                                         min_df=self._min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        else:
            raise ValueError(f'Unknown vectorization type: {self._vectorization}')
        self.vectorizer = vectorizer
        self.sklearn_vector_space = vectorizer.fit_transform(self.data_frame[self._text_col].tolist())
        self.gensim_vector_space = None
        vocab = vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def __iter__(self):
        """A generator that yields a tokenized (i.e., split on whitespace) version of each document.
        Currently used for the topic_model.coherence_w2v_metric.
        """
        for _, doc in self.data_frame[self._text_col].items():
            yield doc.split()

    def export(self, file_path):
        self.data_frame.to_csv(path_or_buf=file_path, sep=self._sep, encoding='utf-8')

    def full_text(self, doc_id: int):
        return self.data_frame.iloc[doc_id][self._full_text_col]

    def title(self, doc_id: int):
        return self.data_frame.iloc[doc_id][self._title_col]

    def date(self, doc_id: int):
        return self.data_frame.iloc[doc_id][self._date_col]

    def author(self, doc_id: int):
        aut_str = str(self.data_frame.iloc[doc_id][self._author_col])
        return aut_str.split(', ')

    def affiliation(self, doc_id: int):
        aff_str = str(self.data_frame.iloc[doc_id][self._affiliation_col])
        return aff_str.split(', ')

    def dataset(self, doc_id: int):
        ds_str = str(self.data_frame.iloc[doc_id][self._dataset_col])
        return ds_str.split(', ')

    def id(self, doc_id: int):
        if self._id_col:
            return str(self.data_frame.iloc[doc_id][self._id_col])
        else:
            return str(doc_id)

    def documents_by_author(self, author: str, year=None):
        aut_doc_ids = []
        if year:
            potential_doc_ids = self.doc_ids_year(year)
        else:
            potential_doc_ids = self.data_frame.index.tolist()
        for doc_id in potential_doc_ids:
            if self.is_author(author, doc_id):
                aut_doc_ids.append(doc_id)
        return aut_doc_ids

    def all_authors(self):
        author_list = []
        for doc_id in self.data_frame.index.tolist():
            author_list.extend(self.author(doc_id))
        return list(set(author_list))

    def is_author(self, author: str, doc_id: int):
        return author in self.author(doc_id)

    def docs_for_word(self, word_id: int, sort: bool = True):
        # get the document indices
        doc_idx = np.where(self.sklearn_vector_space[:, word_id].T.toarray()[0] > 0)[0]
        if sort:
            # sort with strongest association first
            sorted_weight_idx = np.argsort(self.sklearn_vector_space[doc_idx, word_id].T.toarray()[0])[::-1]
            doc_idx = doc_idx[sorted_weight_idx]
        return doc_idx.tolist()

    def doc_ids_year(self, year):
        return self.data_frame.loc[self.data_frame[self._date_col].dt.year == year].index.tolist()

    def word_vector_for_document(self, doc_id=None, normalized=False):
        """Normalized: Divide each document's word weights by the sum of its word weights.
                       Results in the word weights for a document summing to 1.
        """
        if doc_id is None or (isinstance(doc_id, list) and (len(doc_id) == 0)):
            if normalized:
                return np.array(self.sklearn_vector_space / self.sklearn_vector_space.sum(axis=1))
            else:
                return self.sklearn_vector_space.toarray()
        elif isinstance(doc_id, int) or (isinstance(doc_id, list) and (len(doc_id) == 1)):
            if normalized:
                return np.array(self.sklearn_vector_space[doc_id, :] / self.sklearn_vector_space[doc_id, :].sum(axis=1))[0]
            else:
                return self.sklearn_vector_space[doc_id, :].toarray()[0]
        elif isinstance(doc_id, list):
            if normalized:
                return np.array(self.sklearn_vector_space[doc_id, :] / self.sklearn_vector_space[doc_id, :].sum(axis=1))
            else:
                return self.sklearn_vector_space[doc_id, :].toarray()
        else:
            print(f"Unknown dtype '{type(doc_id)}'")

    def word_for_id(self, word_id: int):
        return self.vocabulary.get(word_id)

    def id_for_word(self, word):
        for i, s in self.vocabulary.items():
            if s == word:
                return i
        return -1

    def similar_documents(self, doc_id: int, num_docs: int):
        # compute pairwise similarities
        similarities = np.dot(normalize(self.sklearn_vector_space[doc_id, :], axis=1),
                              normalize(self.sklearn_vector_space, axis=1).T).toarray()[0]
        # exclude self pair
        similarities = [s for s in list(zip(self.data_frame.index.tolist(), similarities)) if s[0] != doc_id]
        # sort by most similar first
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]

    def find_similar_document_pairs(self, threshold: float):
        """find all the similar document pairs above a threshold.
        for a tf vectorized corpus, try threshold=0.83
        for a tfidf vectorized corpus, try threshold=0.88
        """
        # compute pairwise similarities
        similarities = np.dot(normalize(self.sklearn_vector_space, axis=1),
                              normalize(self.sklearn_vector_space, axis=1).T)
        # keep similarities above threshold
        similar_pairs = np.where(similarities.toarray() > threshold)
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
