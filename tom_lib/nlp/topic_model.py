# coding: utf-8
from collections import Counter
from abc import ABCMeta, abstractmethod
import numpy as np
from tom_lib.stats import symmetric_kl, agreement_score
from scipy import spatial, cluster
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
import lda

from tom_lib.structure.corpus import Corpus


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

    def greene_metric(self, min_num_topics=10, step=5, max_num_topics=50, top_n_words=10, tao=10,
                      sample=0.8, beta_loss='frobenius', algorithm='variational', verbose=True):
        """
        Higher is better.

        Implements Greene metric to compute the optimal number of topics. Taken from: How Many Topics?
        Stability Analysis for Topic Models from Greene et al. 2014.
        https://arxiv.org/abs/1404.4606

        :param step:
        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param top_n_words: Top n words for topic to use
        :param tao: Number of sampled models to build
        :return: A list of len (max_num_topics - min_num_topics) with the stability of each tested k
        """
        print('=' * 50)
        print('Computing Greene metric (higher is better)...')
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        stability = []

        # Build reference topic model
        # Generate tao topic models with tao samples of the corpus
        for idx, k in enumerate(num_topics_infer):
            if verbose:
                print(f'Topics={k} ({idx + 1} of {len(num_topics_infer)})')
            if type(self).__name__ == 'NonNegativeMatrixFactorization':
                self.infer_topics(num_topics=k, beta_loss=beta_loss)
            elif type(self).__name__ == 'LatentDirichletAllocation':
                self.infer_topics(num_topics=k, algorithm=algorithm)
            else:
                raise TypeError(f'Unsupported model type: {type(self).__name__}')
            reference_rank = [list(zip(*self.top_words(i, top_n_words)))[0] for i in range(k)]
            agreement_score_list = []
            for t in range(tao):
                tao_corpus = Corpus(
                    source_filepath=self.corpus.data_frame,
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
                if type(self).__name__ == 'NonNegativeMatrixFactorization':
                    tao_model.infer_topics(num_topics=k, beta_loss=beta_loss)
                elif type(self).__name__ == 'LatentDirichletAllocation':
                    tao_model.infer_topics(num_topics=k, algorithm=algorithm)
                else:
                    raise TypeError(f'Unsupported model type: {type(self).__name__}')
                tao_rank = [next(zip(*tao_model.top_words(i, top_n_words))) for i in range(k)]
                agreement_score_list.append(agreement_score(reference_rank, tao_rank))
            stability.append(np.mean(agreement_score_list))
            if verbose:
                print(f'    Stability={stability[-1]:.4f}')
        return stability

    def arun_metric(self, min_num_topics=10, step=5, max_num_topics=50, iterations=10,
                    beta_loss='frobenius', algorithm='variational', verbose=True):
        """
        Lower is better.

        Implements Arun metric to estimate the optimal number of topics:
        Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
        On finding the natural number of topics with latent dirichlet allocation: Some observations.
        In PAKDD (2010), pp. 391–402.
        https://doi.org/10.1007/978-3-642-13657-3_43

        :param min_num_topics: Minimum number of topics to test
        :param max_num_topics: Maximum number of topics to test
        :param iterations: Number of iterations per value of k
        :return: A list of len (max_num_topics - min_num_topics) with the average symmetric KL divergence for each k
        """
        print('=' * 50)
        print('Computing Arun metric (lower is better)...')
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        kl_matrix = []
        for j in range(iterations):
            if verbose:
                print(f'Iteration: {j+1} of {iterations}')
            kl_list = []
            doc_len = np.array([sum(self.corpus.vector_for_document(doc_id)) for doc_id in self.corpus.data_frame.index.tolist()])  # document length
            norm = np.linalg.norm(doc_len)
            for idx, i in enumerate(num_topics_infer):
                if verbose:
                    print(f'    Topics={i} ({idx + 1} of {len(num_topics_infer)})')
                if type(self).__name__ == 'NonNegativeMatrixFactorization':
                    self.infer_topics(num_topics=i, beta_loss=beta_loss)
                elif type(self).__name__ == 'LatentDirichletAllocation':
                    self.infer_topics(num_topics=i, algorithm=algorithm)
                else:
                    raise TypeError(f'Unsupported model type: {type(self).__name__}')
                c_m1 = np.linalg.svd(self.topic_word_matrix.todense(), compute_uv=False)
                c_m2 = doc_len.dot(self.document_topic_matrix.todense())
                c_m2 += 0.0001  # we need this to prevent components equal to zero
                c_m2 /= norm
                kl_list.append(symmetric_kl(c_m1.tolist(), c_m2.tolist()[0]))
            kl_matrix.append(kl_list)
            if verbose:
                print(f'    KL list={kl_list}')
        ouput = np.array(kl_matrix)
        return ouput.mean(axis=0)

    def brunet_metric(self, min_num_topics=10, step=5, max_num_topics=50, iterations=10,
                      beta_loss='frobenius', algorithm='variational', verbose=True):
        """
        Higher is better.

        Implements a consensus-based metric to estimate the optimal number of topics:
        Brunet, J.P., Tamayo, P., Golub, T.R., Mesirov, J.P.
        Metagenes and molecular pattern discovery using matrix factorization.
        Proc. National Academy of Sciences 101(12) (2004), pp. 4164–4169
        https://dx.doi.org/10.1073%2Fpnas.0308531101

        :param min_num_topics:
        :param max_num_topics:
        :param iterations:
        :return:
        """
        print('=' * 50)
        print('Computing Brunet metric (higher is better)...')
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        cophenetic_correlation = []
        for idx, i in enumerate(num_topics_infer):
            if verbose:
                print(f'Topics={i} ({idx + 1} of {len(num_topics_infer)})')
            average_C = np.zeros((self.corpus.size, self.corpus.size))
            for j in range(iterations):
                if verbose:
                    print(f'    Iteration: {j+1} of {iterations}')
                if type(self).__name__ == 'NonNegativeMatrixFactorization':
                    self.infer_topics(num_topics=i, beta_loss=beta_loss)
                elif type(self).__name__ == 'LatentDirichletAllocation':
                    self.infer_topics(num_topics=i, algorithm=algorithm)
                else:
                    raise TypeError(f'Unsupported model type: {type(self).__name__}')
                mlt = np.array([self.most_likely_topic_for_document(doc_id) for doc_id in self.corpus.data_frame.index.tolist()])
                average_C[np.equal(mlt, mlt[:, np.newaxis])] += float(1. / iterations)

            if verbose:
                print('    Clustering...')
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
                print(f'\tCophenetic correlation={c}')
        return cophenetic_correlation

    def perplexity_metric(
        self, min_num_topics=10, step=5, max_num_topics=50,
            train_size=0.7, algorithm='variational', verbose=True):
        """
        Measures perplexity for LDA as computed by scikit-learn.

        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity

        NB: Only supports algorithm='variational' (sklearn LDA)

        :param min_num_topics:
        :param max_num_topics:
        :param train_size:
        :return:
        """
        print('=' * 50)
        print('Computing perplexity metric (lower is better)...')
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        train_perplexities = []
        test_perplexities = []
        if type(self).__name__ == 'LatentDirichletAllocation':
            print(f"Computing perplexity with algorithm='{algorithm}'")
            df_train, df_test = train_test_split(self.corpus.data_frame, train_size=train_size, test_size=1 - train_size)
            corpus_train = Corpus(
                source_filepath=df_train,
                sep=self.corpus._sep,
                language=self.corpus._language,
                n_gram=self.corpus._n_gram,
                vectorization=self.corpus._vectorization,
                max_relative_frequency=self.corpus._max_relative_frequency,
                min_absolute_frequency=self.corpus._min_absolute_frequency,
                max_features=self.corpus.max_features,
                sample=None,
            )
            tf_test = corpus_train.vectorizer.transform(df_test[corpus_train._text_col].tolist())
            lda_model = type(self)(corpus_train)
            for idx, i in enumerate(num_topics_infer):
                if verbose:
                    print(f'Topics={i} ({idx + 1} of {len(num_topics_infer)})')
                lda_model.infer_topics(i, algorithm=algorithm)
                train_perplexities.append(lda_model.model.perplexity(
                    corpus_train.sklearn_vector_space))
                test_perplexities.append(lda_model.model.perplexity(tf_test))
                if verbose:
                    print(f'\tTrain perplexity={train_perplexities[-1]:.4f}, Test perplexity={test_perplexities[-1]:.4f}')
        else:
            raise TypeError("Computing perplexity only supported for LDA with algorithm='variational'. Not running.")
        return train_perplexities, test_perplexities

    def print_topics(self, num_words: int = 10, sort_by_freq: str = ''):
        frequency = self.topics_frequency(count=False)
        frequency_count = self.topics_frequency(count=True)
        topic_list = []
        for topic_id in range(self.nb_topics):
            top_words = [weighted_word[0] for weighted_word in self.top_words(topic_id, num_words)]
            topic_list.append((topic_id, frequency[topic_id], frequency_count[topic_id], top_words))
        if sort_by_freq.lower() == 'asc':
            topic_list.sort(key=lambda x: x[1], reverse=False)
        elif sort_by_freq.lower() == 'desc':
            topic_list.sort(key=lambda x: x[1], reverse=True)
        print('Topic # \tFreq\tCount\tTop words')
        for topic_id, frequency, frequency_count, top_words in topic_list:
            print(f"topic {topic_id:2d}\t{frequency:.4f}\t{int(frequency_count):d}\t{' '.join(top_words)}")

    def top_words(self, topic_id: int, num_words: int):
        word_ids = np.argsort(self.word_distribution_for_topic(topic_id))[:-num_words - 1:-1]
        weighted_words = [(self.corpus.word_for_id(word_id), self.topic_word_matrix[topic_id, word_id]) for word_id in word_ids]
        return weighted_words

    def print_top_docs(self, topics=-1, top_n: int = 10, weights: bool = False, num_words: int = 10):
        for t in list(self.top_topic_docs(topics=topics, top_n=top_n, weights=weights)):
            top_terms = list(self.top_words(t[0], num_words))
            print('=' * 30)
            print(f'Topic {t[0]}: {top_terms}')
            for d in t[1]:
                print('-' * 30)
                if weights:
                    print(f'Document {d[0]}, weight for topic {d[1]:.6f}')
                    doc_id = d[0]
                else:
                    print(f'Document {d}')
                    doc_id = d
                print(self.corpus.affiliation(doc_id), self.corpus.date(doc_id))
                print(f'Title: {self.corpus.title(doc_id)}')
                print(self.corpus.full_text(doc_id))

    def top_topic_docs(self, topics=-1, top_n: int = 10, weights: bool = False):
        '''Inspired by Textacy:
        http://textacy.readthedocs.io/en/latest/_modules/textacy/tm/topic_model.html#TopicModel.top_topic_docs
        '''
        if topics == -1:
            topics = range(self.nb_topics)
        elif isinstance(topics, int):
            topics = (topics,)

        for topic_id in topics:
            doc_ids = np.argsort(self.document_distribution_for_topic(topic_id))[:-top_n - 1:-1]

            if weights is False:
                yield (topic_id,
                       tuple(doc_id for doc_id in doc_ids))
            else:
                yield (topic_id,
                       tuple((doc_id, self.document_topic_matrix[doc_id, topic_id]) for doc_id in doc_ids))

    def top_documents(self, topic_id: int, num_docs: int):
        doc_ids = np.argsort(self.document_distribution_for_topic(topic_id))[:-num_docs - 1:-1]
        weighted_docs = [(doc_id, self.document_topic_matrix[doc_id, topic_id]) for doc_id in doc_ids]
        return weighted_docs

    def word_distribution_for_topic(self, topic_id=None):
        if topic_id is None or (isinstance(topic_id, list) and (len(topic_id) == 0)):
            return self.topic_word_matrix.toarray()
        if isinstance(topic_id, int):
            return self.topic_word_matrix[topic_id, :].toarray()[0]
        elif isinstance(topic_id, list):
            return self.topic_word_matrix[topic_id, :].toarray()
        else:
            print(f"Unknown dtype '{type(topic_id)}'")

    def document_distribution_for_topic(self, topic_id=None):
        if topic_id is None or (isinstance(topic_id, list) and (len(topic_id) == 0)):
            return self.document_topic_matrix.toarray()
        if isinstance(topic_id, int):
            return self.document_topic_matrix[:, topic_id].toarray().T[0]
        elif isinstance(topic_id, list):
            return self.document_topic_matrix[:, topic_id].toarray().T
        else:
            print(f"Unknown dtype '{type(topic_id)}'")

    def topic_distribution_for_document(self, doc_id=None):
        if doc_id is None or (isinstance(doc_id, list) and (len(doc_id) == 0)):
            return self.document_topic_matrix.toarray()
        if isinstance(doc_id, int):
            return self.document_topic_matrix[doc_id, :].toarray()[0]
        elif isinstance(doc_id, list):
            return self.document_topic_matrix[doc_id, :].toarray()
        else:
            print(f"Unknown dtype '{type(doc_id)}'")

    def topic_distribution_for_word(self, word_id=None):
        if word_id is None or (isinstance(word_id, list) and (len(word_id) == 0)):
            return self.topic_word_matrix.toarray()
        if isinstance(word_id, int):
            return self.topic_word_matrix[:, word_id].toarray().T[0]
        elif isinstance(word_id, list):
            return self.topic_word_matrix[:, word_id].toarray().T
        else:
            print(f"Unknown dtype '{type(word_id)}'")

    def topic_distribution_for_author(self, author_name: str):
        all_weights = []
        for document_id in self.corpus.documents_by_author(author_name):
            all_weights.append(self.topic_distribution_for_document(document_id))
        output = np.array(all_weights)
        return output.mean(axis=0)

    def most_likely_topic_for_document(self, doc_id: int = None):
        if doc_id is None or (isinstance(doc_id, list) and (len(doc_id) == 0)):
            return np.argmax(self.document_topic_matrix.toarray(), axis=1)
        if isinstance(doc_id, int):
            return np.argmax(self.topic_distribution_for_document(doc_id), axis=0)
        elif isinstance(doc_id, list):
            return np.argmax(self.topic_distribution_for_document(doc_id), axis=1)
        else:
            print(f"Unknown dtype '{type(doc_id)}'")

    def topic_frequency(self, topic, date=None, count: bool = False):
        return self.topics_frequency(date=date, count=count)[topic]

    def topics_frequency(self, date=None, count: bool = False):
        if date is None:
            doc_ids = self.corpus.data_frame.index.tolist()
        else:
            doc_ids = self.corpus.doc_ids(date)

        topic_count = Counter(self.most_likely_topic_for_document(doc_ids))
        frequency = np.array([topic_count[i] if i in topic_count else 0 for i in range(self.nb_topics)])

        if not count:
            frequency = frequency / len(doc_ids)

        return frequency

    def documents_for_topic(self, topic_id: int):
        return self.corpus.data_frame.index[self.most_likely_topic_for_document() == topic_id].tolist()

    def documents_per_topic(self):
        return {i: self.documents_for_topic(i) for i in range(self.nb_topics)}

    def affiliation_repartition(self, topic_id: int):
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
        self.algorithm = algorithm
        lda_model = None
        topic_document = None
        if self.algorithm == 'variational':
            lda_model = LDA(n_components=num_topics, learning_method='batch')
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        elif self.algorithm == 'gibbs':
            lda_model = lda.LDA(n_topics=num_topics, n_iter=500)
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        else:
            raise ValueError(f"algorithm must be either 'variational' or 'gibbs', got {self.algorithm}")
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
    def infer_topics(self, num_topics=10, beta_loss='frobenius', **kwargs):
        self.nb_topics = num_topics
        self.beta_loss = beta_loss
        if self.beta_loss not in ['frobenius', 'kullback-leibler', 'itakura-saito']:
            raise ValueError(f"beta_loss must be 'frobenius', 'kullback-leibler', or 'itakura-saito', got {self.beta_loss}")
        nmf_model = NMF(n_components=num_topics, beta_loss=self.beta_loss)
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
