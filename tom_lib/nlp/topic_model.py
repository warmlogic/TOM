# coding: utf-8
import itertools
from abc import ABCMeta, abstractmethod
import numpy as np
import tom_lib.stats
from scipy import spatial, cluster
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
import lda

from tom_lib.structure.corpus import Corpus

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus  # a Corpus object
        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None  # a scalar value > 1
        self._sample = corpus._sample
        self.model = None

    @abstractmethod
    def infer_topics(self, num_topics=10, **kwargs):
        pass

    def greene_metric(self, min_num_topics=10, step=5, max_num_topics=50, top_n_words=10, tao=10, verbose=True):
        """
        Implements Greene metric to compute the optimal number of topics. Taken from How Many Topics?
        Stability Analysis for Topic Models from Greene et al. 2014.
        :param step:
        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param top_n_words: Top n words for topic to use
        :param tao: Number of sampled models to build
        :return: A list of len (max_num_topics - min_num_topics) with the stability of each tested k
        """
        if self._sample:
            sample = self._sample
        else:
            sample = True

        stability = []
        # Build reference topic model
        # Generate tao topic models with tao samples of the corpus

        for k in range(min_num_topics, max_num_topics + 1, step):
            if verbose:
                print('Topics={}'.format(k))
            self.infer_topics(k)
            reference_rank = [list(zip(*self.top_words(i, top_n_words)))[0] for i in range(k)]
            agreement_score_list = []
            for t in range(tao):
                tao_corpus = Corpus(
                    source_file_path=self.corpus.data_frame,
                    sep=self.corpus._sep,
                    language=self.corpus._language,
                    n_gram=self.corpus._n_gram,
                    vectorization=self.corpus._vectorization,
                    max_relative_frequency=self.corpus._max_relative_frequency,
                    min_absolute_frequency=self.corpus._min_absolute_frequency,
                    max_features=self.corpus.max_features,
                    sample=sample,
                    )
                tao_model = type(self)(tao_corpus)
                tao_model.infer_topics(k)
                tao_rank = [next(zip(*tao_model.top_words(i, top_n_words))) for i in range(k)]
                agreement_score_list.append(tom_lib.stats.agreement_score(reference_rank, tao_rank))
            stability.append(np.mean(agreement_score_list))
            if verbose:
                print('\tStability={:.4f}'.format(k, stability[-1]))
        return stability

    def arun_metric(self, min_num_topics=10, step=5, max_num_topics=50, iterations=10, verbose=True):
        """
        Implements Arun metric to estimate the optimal number of topics:
        Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
        On finding the natural number of topics with latent dirichlet allocation: Some observations.
        In PAKDD (2010), pp. 391–402.
        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param iterations: Number of iterations per value of k
        :return: A list of len (max_num_topics - min_num_topics) with the average symmetric KL divergence for each k
        """
        kl_matrix = []
        for j in range(iterations):
            if verbose:
                print('Iteration={}'.format(j))
            kl_list = []
            l = np.array([sum(self.corpus.vector_for_document(doc_id)) for doc_id in range(self.corpus.size)])  # document length
            norm = np.linalg.norm(l)
            for i in range(min_num_topics, max_num_topics + 1, step):
                self.infer_topics(i)
                c_m1 = np.linalg.svd(self.topic_word_matrix.todense(), compute_uv=False)
                c_m2 = l.dot(self.document_topic_matrix.todense())
                c_m2 += 0.0001  # we need this to prevent components equal to zero
                c_m2 /= norm
                kl_list.append(tom_lib.stats.symmetric_kl(c_m1.tolist(), c_m2.tolist()[0]))
            kl_matrix.append(kl_list)
            if verbose:
                print('\tKL list={}'.format(j, kl_list))
        ouput = np.array(kl_matrix)
        return ouput.mean(axis=0)

    def brunet_metric(self, min_num_topics=10, step=5, max_num_topics=50, iterations=10, verbose=True):
        """
        Implements a consensus-based metric to estimate the optimal number of topics:
        Brunet, J.P., Tamayo, P., Golub, T.R., Mesirov, J.P.
        Metagenes and molecular pattern discovery using matrix factorization.
        Proc. National Academy of Sciences 101(12) (2004), pp. 4164–4169
        :param min_num_topics:
        :param max_num_topics:
        :param iterations:
        :return:
        """
        cophenetic_correlation = []
        for i in range(min_num_topics, max_num_topics + 1, step):
            if verbose:
                print('Topics={}'.format(i))
            average_C = np.zeros((self.corpus.size, self.corpus.size))
            for j in range(iterations):
                self.infer_topics(i)
                for p in range(self.corpus.size):
                    for q in range(self.corpus.size):
                        if self.most_likely_topic_for_document(p) == self.most_likely_topic_for_document(q):
                            average_C[p, q] += float(1./iterations)

            clustering = cluster.hierarchy.linkage(average_C, method='average')
            Z = cluster.hierarchy.dendrogram(clustering, orientation='right')
            index = Z['leaves']
            average_C = average_C[index, :]
            average_C = average_C[:, index]
            (c, d) = cluster.hierarchy.cophenet(Z=clustering, Y=spatial.distance.pdist(average_C))
            # plt.clf()
            # f, ax = plt.subplots(figsize=(11, 9))
            # ax = sns.heatmap(average_C)
            # plt.savefig('reorderedC.png')
            cophenetic_correlation.append(c)
            if verbose:
                print('\tCophenetic correlation={}'.format(i, c))
        return cophenetic_correlation

    def perplexity_metric(self, min_num_topics=10, step=5, max_num_topics=50,
        train_size=0.7, verbose=True):
        """
        Measures perplexity for LDA as computed by scikit-learn.

        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity

        NB: Only supports algorithm='variational' (sklearn LDA)

        :param min_num_topics:
        :param max_num_topics:
        :param train_size:
        :return:
        """

        train_perplexities = []
        test_perplexities = []
        if isinstance(self, LatentDirichletAllocation):
            algorithm = 'variational'
            print("Computing perplexity with algorithm='{}'".format(algorithm))
            df_train, df_test = train_test_split(self.corpus.data_frame, train_size=train_size, test_size=1-train_size)
            corpus_train = Corpus(
                source_file_path=df_train,
                sep=self.corpus._sep,
                language=self.corpus._language,
                n_gram=self.corpus._n_gram,
                vectorization=self.corpus._vectorization,
                max_relative_frequency=self.corpus._max_relative_frequency,
                min_absolute_frequency=self.corpus._min_absolute_frequency,
                max_features=self.corpus.max_features,
                sample=None,
                )
            tf_test = corpus_train.vectorizer.transform(df_test['text'].tolist())
            lda_model = type(self)(corpus_train)
            for i in range(min_num_topics, max_num_topics + 1, step):
                if verbose:
                    print('Topics={}'.format(i))
                lda_model.infer_topics(i, algorithm=algorithm)
                train_perplexities.append(lda_model.model.perplexity(
                    corpus_train.sklearn_vector_space))
                test_perplexities.append(lda_model.model.perplexity(tf_test))
                if verbose:
                    print('\tTrain perplexity={:.4f}, Test perplexity={:.4f}'.format(
                        train_perplexities[-1], test_perplexities[-1]))
        else:
            raise TypeError("Computing perplexity only supported for LDA with algorithm='variational'. Not running.")
        return train_perplexities, test_perplexities

    def print_topics(self, num_words=10, sort_by_freq=''):
        frequency = self.topics_frequency()
        topic_list = []
        for topic_id in range(self.nb_topics):
            word_list = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
            topic_list.append((topic_id, frequency[topic_id], word_list))
        if sort_by_freq == 'asc':
            topic_list.sort(key=lambda x: x[1], reverse=False)
        elif sort_by_freq == 'desc':
            topic_list.sort(key=lambda x: x[1], reverse=True)
        for topic_id, frequency, topic_desc in topic_list:
            print('topic %d\t%f\t%s' % (topic_id, frequency, ' '.join(topic_desc)))

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1], reverse=True)
        return weighted_words[:num_words]

    def print_top_docs(self, topics, top_n, weights=False, num_words=10):
        for t in list(self.top_topic_docs(topics=topics, top_n=top_n, weights=weights)):
            top_terms = list(self.top_words(t[0], num_words))
            print('='*30)
            print('Topic {}: {}'.format(t[0], top_terms))
            for d in t[1]:
                print('-'*30)
                print('Document {}'.format(d))
                print(self.corpus.affiliation(d), self.corpus.date(d))
                print('Title:', self.corpus.title(d))
                print(self.corpus.full_text(d))

    def top_topic_docs(self, topics=-1, top_n=10, weights=False):
        '''Inspired by Textacy:
        http://textacy.readthedocs.io/en/latest/_modules/textacy/tm/topic_model.html#TopicModel.top_topic_docs
        '''
        if topics == -1:
            topics = range(self.nb_topics)
        elif isinstance(topics, int):
            topics = (topics,)

        for topic_idx in topics:
            top_doc_idxs = np.argsort(self.document_topic_matrix[:, topic_idx].toarray(), axis=0)[:-top_n - 1:-1]
            top_doc_idxs = [i[0] for i in top_doc_idxs]

            if weights is False:
                yield (topic_idx,
                       tuple(doc_idx for doc_idx in top_doc_idxs))
            else:
                yield (topic_idx,
                       tuple((doc_idx, self.document_topic_matrix[doc_idx, topic_idx]) for doc_idx in top_doc_idxs))

    def word_distribution_for_topic(self, topic_id):
        vector = self.topic_word_matrix[topic_id].toarray()
        return vector[0]

    def topic_distribution_for_document(self, doc_id):
        vector = self.document_topic_matrix[doc_id].toarray()
        return vector[0]

    def topic_distribution_for_word(self, word_id):
        vector = self.topic_word_matrix[:, word_id].toarray()
        return vector.T[0]

    def topic_distribution_for_author(self, author_name):
        all_weights = []
        for document_id in self.corpus.documents_by_author(author_name):
            all_weights.append(self.topic_distribution_for_document(document_id))
        output = np.array(all_weights)
        return output.mean(axis=0)

    def most_likely_topic_for_document(self, doc_id):
        weights = list(self.topic_distribution_for_document(doc_id))
        return weights.index(max(weights))

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = np.zeros(self.nb_topics)
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date)
        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0 / len(ids)
        return frequency

    def documents_for_topic(self, topic_id):
        doc_ids = []
        for doc_id in range(self.corpus.size):
            most_likely_topic = self.most_likely_topic_for_document(doc_id)
            if most_likely_topic == topic_id:
                doc_ids.append(doc_id)
        return doc_ids

    def documents_per_topic(self):
        topic_associations = {}
        for i in range(self.corpus.size):
            topic_id = self.most_likely_topic_for_document(i)
            if topic_associations.get(topic_id):
                documents = topic_associations[topic_id]
                documents.append(i)
                topic_associations[topic_id] = documents
            else:
                documents = [i]
                topic_associations[topic_id] = documents
        return topic_associations

    def affiliation_repartition(self, topic_id):
        counts = {}
        doc_ids = self.documents_for_topic(topic_id)
        for i in doc_ids:
            affiliations = set(self.corpus.affiliation(i))
            for affiliation in affiliations:
                if counts.get(affiliation) is not None:
                    count = counts[affiliation] + 1
                    counts[affiliation] = count
                else:
                    counts[affiliation] = 1
        tuples = []
        for affiliation, count in counts.items():
            tuples.append((affiliation, count))
        tuples.sort(key=lambda x: x[1], reverse=True)
        return tuples

    def topic_distribution_for_new_document(self, text):
        doc_topic_distr = self.model.transform(
            self.corpus.vectorizer.transform([text]))[0]
        return doc_topic_distr

class LatentDirichletAllocation(TopicModel):
    def infer_topics(self, num_topics=10, algorithm='variational', **kwargs):
        self.nb_topics = num_topics
        lda_model = None
        topic_document = None
        if algorithm == 'variational':
            lda_model = LDA(n_components=num_topics, learning_method='batch')
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        elif algorithm == 'gibbs':
            lda_model = lda.LDA(n_topics=num_topics, n_iter=500)
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        else:
            raise ValueError("algorithm must be either 'variational' or 'gibbs', got {}".format(algorithm))
        # store the model for future use
        self.model = lda_model
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(lda_model.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
            doc_count += 1
        self.document_topic_matrix = coo_matrix((data, (row, col)),
                                                shape=(self.corpus.size, self.nb_topics)).tocsr()


class NonNegativeMatrixFactorization(TopicModel):
    def infer_topics(self, num_topics=10, **kwargs):
        self.nb_topics = num_topics
        nmf_model = NMF(n_components=num_topics)
        topic_document = nmf_model.fit_transform(self.corpus.sklearn_vector_space)
        # store the model for future use
        self.model = nmf_model
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
            doc_count += 1
        self.document_topic_matrix = coo_matrix((data, (row, col)),
                                                shape=(self.corpus.size, self.nb_topics)).tocsr()
