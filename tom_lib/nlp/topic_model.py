# coding: utf-8
from collections import Counter
from abc import ABCMeta, abstractmethod
import numpy as np
from tom_lib.stats import symmetric_kl, agreement_score
from scipy import spatial, cluster
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import lda  # https://github.com/ariddell/lda/

from tom_lib.structure.corpus import Corpus

from itertools import combinations
from gensim.models import Word2Vec
from multiprocessing import cpu_count


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus  # a Corpus object
        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None  # a scalar value > 1
        self._sample = corpus._sample
        self.model = None
        self.model_type = None
        self.random_state = None

    @abstractmethod
    def infer_topics(self, num_topics=10, **kwargs):
        pass

    def greene_metric(
        self,
        min_num_topics: int = 10,
        max_num_topics: int = 20,
        step: int = 5,
        top_n_words: int = 10,
        tao: int = 10,
        sample: float = 0.8,
        verbose: bool = True,
        nmf_init: str = None,
        nmf_solver: str = None,
        nmf_beta_loss: str = None,
        nmf_max_iter: int = None,
        nmf_alpha: float = None,
        nmf_l1_ratio: float = None,
        nmf_shuffle: bool = None,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
    ):
        """
        Higher is better.

        Greene, D., O'Callaghan, D., and Cunningham, P.
        How Many Topics? Stability Analysis for Topic Models
        Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2014.
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
            if self.model_type == 'NMF':
                self.infer_topics(
                    num_topics=k,
                    nmf_init=nmf_init,
                    nmf_solver=nmf_solver,
                    nmf_beta_loss=nmf_beta_loss,
                    nmf_max_iter=nmf_max_iter,
                    nmf_alpha=nmf_alpha,
                    nmf_l1_ratio=nmf_l1_ratio,
                    nmf_shuffle=nmf_shuffle,
                    random_state=random_state,
                )
            elif self.model_type == 'LDA':
                self.infer_topics(
                    num_topics=k,
                    lda_algorithm=lda_algorithm,
                    lda_alpha=lda_alpha,
                    lda_eta=lda_eta,
                    lda_learning_method=lda_learning_method,
                    lda_n_jobs=lda_n_jobs,
                    lda_n_iter=lda_n_iter,
                    random_state=random_state,
                )
            else:
                raise TypeError(f'Unsupported model type: {self.model_type}')
            reference_rank = [list(zip(*self.top_words(i, top_n_words)))[0] for i in range(k)]
            agreement_score_list = []
            for t in range(tao):
                tao_corpus = Corpus(
                    source_filepath=self.corpus.data_frame,
                    name=self.corpus.name,
                    sep=self.corpus._sep,
                    language=self.corpus._language,
                    n_gram=self.corpus._n_gram,
                    vectorization=self.corpus._vectorization,
                    max_relative_frequency=self.corpus._max_relative_frequency,
                    min_absolute_frequency=self.corpus._min_absolute_frequency,
                    max_features=self.corpus.max_features,
                    sample=sample,
                    text_col=self.corpus.text_col,
                    full_text_col=self.corpus.full_text_col,
                    title_col=self.corpus.title_col,
                    author_col=self.corpus.author_col,
                    affiliation_col=self.corpus.affiliation_col,
                    dataset_col=self.corpus.dataset_col,
                    date_col=self.corpus.date_col,
                    id_col=self.corpus.id_col,
                )
                tao_model = type(self)(tao_corpus)
                if self.model_type == 'NMF':
                    tao_model.infer_topics(
                        num_topics=k,
                        nmf_beta_loss=nmf_beta_loss,
                        random_state=random_state,
                    )
                elif self.model_type == 'LDA':
                    tao_model.infer_topics(
                        num_topics=k,
                        lda_algorithm=lda_algorithm,
                        random_state=random_state,
                    )
                else:
                    raise TypeError(f'Unsupported model type: {self.model_type}')
                tao_rank = [next(zip(*tao_model.top_words(i, top_n_words))) for i in range(k)]
                agreement_score_list.append(agreement_score(reference_rank, tao_rank))
            stability.append(np.mean(agreement_score_list))
            if verbose:
                print(f'    Stability={stability[-1]:.4f}')
        return stability

    def arun_metric(
        self,
        min_num_topics: int = 10,
        max_num_topics: int = 20,
        step: int = 5,
        iterations: int = 10,
        verbose: bool = True,
        nmf_init: str = None,
        nmf_solver: str = None,
        nmf_beta_loss: str = None,
        nmf_max_iter: int = None,
        nmf_alpha: float = None,
        nmf_l1_ratio: float = None,
        nmf_shuffle: bool = None,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
    ):
        """
        Lower is better.

        Arun, R., Suresh, V., Madhavan, C. E. V., and Murthy, M. N.
        On finding the natural number of topics with latent dirichlet allocation: Some observations.
        PAKDD (2010), pp. 391–402.
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
            doc_len = self.corpus.word_vector_for_document().sum(axis=1)  # document length
            norm = np.linalg.norm(doc_len)
            for idx, i in enumerate(num_topics_infer):
                if verbose:
                    print(f'    Topics={i} ({idx + 1} of {len(num_topics_infer)})')
                if self.model_type == 'NMF':
                    self.infer_topics(
                        num_topics=i,
                        nmf_init=nmf_init,
                        nmf_solver=nmf_solver,
                        nmf_beta_loss=nmf_beta_loss,
                        nmf_max_iter=nmf_max_iter,
                        nmf_alpha=nmf_alpha,
                        nmf_l1_ratio=nmf_l1_ratio,
                        nmf_shuffle=nmf_shuffle,
                        lda_algorithm=lda_algorithm,
                        lda_alpha=lda_alpha,
                        lda_eta=lda_eta,
                        lda_learning_method=lda_learning_method,
                        lda_n_jobs=lda_n_jobs,
                        lda_n_iter=lda_n_iter,
                        random_state=random_state,
                    )
                elif self.model_type == 'LDA':
                    self.infer_topics(
                        num_topics=i,
                        lda_algorithm=lda_algorithm,
                        lda_alpha=lda_alpha,
                        lda_eta=lda_eta,
                        lda_learning_method=lda_learning_method,
                        lda_n_jobs=lda_n_jobs,
                        lda_n_iter=lda_n_iter,
                        random_state=random_state,
                    )
                else:
                    raise TypeError(f'Unsupported model type: {self.model_type}')
                c_m1 = np.linalg.svd(self.topic_word_matrix.todense(), compute_uv=False)
                c_m2 = doc_len.dot(self.document_topic_matrix.todense())
                c_m2 += 0.0001  # we need this to prevent components equal to zero
                c_m2 /= norm
                kl_list.append(symmetric_kl(c_m1.tolist(), c_m2.tolist()[0]))
            kl_matrix.append(kl_list)
            if verbose:
                print(f'    Iteration KL list={kl_list}')
                print(f'    Iteration Average={np.mean(kl_list)}')
        avg_kl_matrix = np.array(kl_matrix).mean(axis=0).tolist()
        if verbose:
            print(f'            Overall KL average={avg_kl_matrix}')
        return avg_kl_matrix

    def brunet_metric(
        self,
        min_num_topics: int = 10,
        max_num_topics: int = 20,
        step: int = 5,
        iterations: int = 10,
        verbose: bool = True,
        nmf_init: str = None,
        nmf_solver: str = None,
        nmf_beta_loss: str = None,
        nmf_max_iter: int = None,
        nmf_alpha: float = None,
        nmf_l1_ratio: float = None,
        nmf_shuffle: bool = None,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
    ):
        """
        Higher is better.

        Implements a consensus-based metric to estimate the optimal number of topics:
        Brunet, J. P., Tamayo, P., Golub, T. R., and Mesirov, J. P.
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
                if self.model_type == 'NMF':
                    self.infer_topics(
                        num_topics=i,
                        nmf_init=nmf_init,
                        nmf_solver=nmf_solver,
                        nmf_beta_loss=nmf_beta_loss,
                        nmf_max_iter=nmf_max_iter,
                        nmf_alpha=nmf_alpha,
                        nmf_l1_ratio=nmf_l1_ratio,
                        nmf_shuffle=nmf_shuffle,
                        random_state=random_state,
                    )
                elif self.model_type == 'LDA':
                    self.infer_topics(
                        num_topics=i,
                        nmf_init=nmf_init,
                        nmf_solver=nmf_solver,
                        nmf_beta_loss=nmf_beta_loss,
                        nmf_max_iter=nmf_max_iter,
                        nmf_alpha=nmf_alpha,
                        nmf_l1_ratio=nmf_l1_ratio,
                        nmf_shuffle=nmf_shuffle,
                        lda_algorithm=lda_algorithm,
                        lda_alpha=lda_alpha,
                        lda_eta=lda_eta,
                        lda_learning_method=lda_learning_method,
                        lda_n_jobs=lda_n_jobs,
                        lda_n_iter=lda_n_iter,
                        random_state=random_state,
                    )
                else:
                    raise TypeError(f'Unsupported model type: {self.model_type}')
                mlt = self.most_likely_topic_for_document()
                average_C[np.equal(mlt, mlt[:, np.newaxis])] += float(1. / iterations)

            if verbose:
                print('    Clustering...')
            Z = cluster.hierarchy.linkage(average_C, method='average')
            if verbose:
                print('    Getting list of leaves...')
            lvs = cluster.hierarchy.leaves_list(Z)
            average_C = average_C[lvs, :]
            average_C = average_C[:, lvs]
            if verbose:
                print('    Calculating cophenetic distances...')
            (c, d) = cluster.hierarchy.cophenet(Z=Z, Y=spatial.distance.pdist(average_C))
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(11, 9))
            # ax = sns.heatmap(average_C, ax=ax)
            # plt.savefig('reorderedC.png')
            cophenetic_correlation.append(c)
            if verbose:
                print(f'    Cophenetic correlation={c}')
        return cophenetic_correlation

    def coherence_w2v_metric(
        self,
        min_num_topics: int = 10,
        max_num_topics: int = 20,
        step: int = 5,
        top_n_words: int = 10,
        w2v_size: int = None,
        w2v_min_count: int = None,
        # w2v_max_vocab_size: int = None,
        w2v_max_final_vocab: int = None,
        w2v_sg: int = None,
        w2v_workers: int = None,
        verbose: bool = True,
        nmf_init: str = None,
        nmf_solver: str = None,
        nmf_beta_loss: str = None,
        nmf_max_iter: int = None,
        nmf_alpha: float = None,
        nmf_l1_ratio: float = None,
        nmf_shuffle: bool = None,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
    ):
        """
        Higher is better.

        1. Trains a Word2Vec word embedding model on the corpus.
        2. For each topic, for all pairs of its top ranked words, compute their similarity using the Word2Vec model.
           A topic's coherence score is the average of the word pair similarity scores.
        3. A topic model's coherence score is the grand average: the average of the topic coherence scores.

        Adapted from https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
        """

        def calculate_coherence(w2v_model, top_words_topics):
            overall_coherence = 0.0
            for tid in range(self.nb_topics):
                # check each pair of terms
                pair_scores = []
                for pair in combinations(top_words_topics[tid], 2):
                    try:
                        score = w2v_model.wv.similarity(pair[0], pair[1])
                    except KeyError:
                        # one of the words is not in w2v_model.wv.vocab
                        score = 0
                    pair_scores.append(score)
                # get the mean for all pairs in this topic
                topic_score = sum(pair_scores) / len(pair_scores)
                overall_coherence += topic_score
            # get the mean score across all topics
            return overall_coherence / self.nb_topics

        w2v_size = w2v_size or 100
        w2v_min_count = w2v_min_count or self.corpus._min_absolute_frequency
        # w2v_max_vocab_size = w2v_max_vocab_size or self.corpus.max_features
        w2v_max_final_vocab = w2v_max_final_vocab or self.corpus.max_features
        w2v_sg = w2v_sg or 1
        w2v_workers = w2v_workers or cpu_count() - 1

        print('=' * 50)
        print('Computing coherence Word2Vec metric (higher is better)...')

        print('Step 1/2: Training Word2Vec model...')
        w2v_model = Word2Vec(
            self.corpus,
            size=w2v_size,
            min_count=w2v_min_count,
            # max_vocab_size=w2v_max_vocab_size,
            max_final_vocab=w2v_max_final_vocab,
            sg=w2v_sg,
            workers=w2v_workers,
        )
        if verbose:
            print(f'    Word2Vec model has {len(w2v_model.wv.vocab)} terms')

        print('Step 2/2: Training topic models...')
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        coherence = []
        for idx, k in enumerate(num_topics_infer):
            if verbose:
                print(f'Topics={k} ({idx + 1} of {len(num_topics_infer)})')
            if self.model_type == 'NMF':
                self.infer_topics(
                    num_topics=k,
                    nmf_init=nmf_init,
                    nmf_solver=nmf_solver,
                    nmf_beta_loss=nmf_beta_loss,
                    nmf_max_iter=nmf_max_iter,
                    nmf_alpha=nmf_alpha,
                    nmf_l1_ratio=nmf_l1_ratio,
                    nmf_shuffle=nmf_shuffle,
                    random_state=random_state,
                )
            elif self.model_type == 'LDA':
                self.infer_topics(
                    num_topics=k,
                    lda_algorithm=lda_algorithm,
                    lda_alpha=lda_alpha,
                    lda_eta=lda_eta,
                    lda_learning_method=lda_learning_method,
                    lda_n_jobs=lda_n_jobs,
                    lda_n_iter=lda_n_iter,
                    random_state=random_state,
                )
            else:
                raise TypeError(f'Unsupported model type: {self.model_type}')

            top_words_topics = self.top_words_topics(num_words=top_n_words)
            coh = calculate_coherence(w2v_model, top_words_topics)
            if verbose:
                print(f'    Coherence: {coh:.5f}')
            coherence.append(coh)

        return coherence

    def perplexity_metric(
        self,
        min_num_topics: int = 10,
        max_num_topics: int = 20,
        step: int = 5,
        train_size: float = 0.7,
        verbose: bool = True,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
    ):
        """
        Measures perplexity for LDA as computed by scikit-learn.

        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity

        NB: Only supports lda_algorithm: str = None (sklearn LDA)

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
        if self.model_type == 'LDA':
            print(f"Computing perplexity with lda_algorithm='{lda_algorithm}'")
            df_train, df_test = train_test_split(self.corpus.data_frame, train_size=train_size, test_size=1 - train_size)
            corpus_train = Corpus(
                source_filepath=df_train,
                name=self.corpus.name,
                sep=self.corpus._sep,
                language=self.corpus._language,
                n_gram=self.corpus._n_gram,
                vectorization=self.corpus._vectorization,
                max_relative_frequency=self.corpus._max_relative_frequency,
                min_absolute_frequency=self.corpus._min_absolute_frequency,
                max_features=self.corpus.max_features,
                sample=None,
                text_col=self.corpus.text_col,
                full_text_col=self.corpus.full_text_col,
                title_col=self.corpus.title_col,
                author_col=self.corpus.author_col,
                affiliation_col=self.corpus.affiliation_col,
                dataset_col=self.corpus.dataset_col,
                date_col=self.corpus.date_col,
                id_col=self.corpus.id_col,
            )
            tf_test = corpus_train.vectorizer.transform(df_test[corpus_train._text_col].tolist())
            lda_model = type(self)(corpus_train)
            for idx, i in enumerate(num_topics_infer):
                if verbose:
                    print(f'Topics={i} ({idx + 1} of {len(num_topics_infer)})')
                lda_model.infer_topics(
                    num_topics=i,
                    lda_algorithm=lda_algorithm,
                    lda_alpha=lda_alpha,
                    lda_eta=lda_eta,
                    lda_learning_method=lda_learning_method,
                    lda_n_jobs=lda_n_jobs,
                    lda_n_iter=lda_n_iter,
                    random_state=random_state,
                )
                train_perplexities.append(lda_model.model.perplexity(
                    corpus_train.sklearn_vector_space))
                test_perplexities.append(lda_model.model.perplexity(tf_test))
                if verbose:
                    print(f'\tTrain perplexity={train_perplexities[-1]:.4f}, Test perplexity={test_perplexities[-1]:.4f}')
        else:
            raise TypeError("Computing perplexity only supported for LDA with lda_algorithm: str = None. Not running.")
        return train_perplexities, test_perplexities

    def print_topics(self, num_words: int = 10, sort_by_freq: str = ''):
        frequency = self.topics_frequency(count=False)
        frequency_count = self.topics_frequency(count=True)
        top_words_topics = self.top_words_topics(num_words)
        topic_list = []
        for topic_id in range(self.nb_topics):
            topic_list.append((topic_id, frequency[topic_id], frequency_count[topic_id], top_words_topics[topic_id]))
        if sort_by_freq.lower() == 'asc':
            topic_list.sort(key=lambda x: x[1], reverse=False)
        elif sort_by_freq.lower() == 'desc':
            topic_list.sort(key=lambda x: x[1], reverse=True)
        print('Topic # \tFreq\tCount\tTop words')
        for topic_id, frequency, frequency_count, top_words in topic_list:
            print(f"topic {topic_id:2d}\t{frequency:.4f}\t{int(frequency_count):d}\t{' '.join(top_words)}")

    def top_words(self, topic_id: int, normalized: bool = True, num_words: int = 10):
        weights = self.word_distribution_for_topic(topic_id=topic_id, normalized=normalized)
        word_ids = np.argsort(weights)[:-num_words - 1:-1]
        weighted_words = [(self.corpus.word_for_id(word_id), weights[word_id]) for word_id in word_ids]
        return weighted_words

    def top_words_topics(self, num_words: int = 10):
        top_words_topics = []
        for tid in range(self.nb_topics):
            top_words = [weighted_word[0] for weighted_word in self.top_words(topic_id=tid, num_words=num_words)]
            top_words_topics.append(top_words)
        return top_words_topics

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
        """Inspired by Textacy:
        http://textacy.readthedocs.io/en/latest/_modules/textacy/tm/topic_model.html#TopicModel.top_topic_docs
        """
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

    def word_distribution_for_topic(self, topic_id=None, normalized=False):
        """Normalized: Divide each topic's word weights by the sum of its word weights.
                       Results in the word weights for a topic summing to 1.
        """
        if topic_id is None or (isinstance(topic_id, list) and (len(topic_id) == 0)):
            # all topics
            if normalized:
                return self.topic_word_matrix / self.topic_word_matrix.sum(axis=1)
            else:
                return self.topic_word_matrix.toarray()
        elif isinstance(topic_id, int) or (isinstance(topic_id, list) and (len(topic_id) == 1)):
            # one topic
            if normalized:
                return np.array(self.topic_word_matrix[topic_id, :] / self.topic_word_matrix[topic_id, :].sum(axis=1))[0]
            else:
                return self.topic_word_matrix[topic_id, :].toarray()[0]
        elif isinstance(topic_id, list):
            # list of topics
            if normalized:
                return np.array(self.topic_word_matrix[topic_id, :] / self.topic_word_matrix[topic_id, :].sum(axis=1))
            else:
                return self.topic_word_matrix[topic_id, :].toarray()
        else:
            print(f"Unknown dtype '{type(topic_id)}'")

    def document_distribution_for_topic(self, topic_id=None, normalized=False):
        """Normalized: Divide each topic's document weights by the sum of its document weights.
                       Results in the document weights for a topic summing to 1.
        """
        if topic_id is None or (isinstance(topic_id, list) and (len(topic_id) == 0)):
            # all topics
            if normalized:
                return self.document_topic_matrix / self.document_topic_matrix.sum(axis=0)
            else:
                return self.document_topic_matrix.toarray()
        elif isinstance(topic_id, int) or (isinstance(topic_id, list) and (len(topic_id) == 1)):
            # one topic
            if normalized:
                return np.array(
                    self.document_topic_matrix[:, topic_id] / self.document_topic_matrix[:, topic_id].sum(axis=0)).T[0]
            else:
                return self.document_topic_matrix[:, topic_id].toarray().T[0]
        elif isinstance(topic_id, list):
            # list of topics
            if normalized:
                return np.array(
                    self.document_topic_matrix[:, topic_id] / self.document_topic_matrix[:, topic_id].sum(axis=0)).T
            else:
                return self.document_topic_matrix[:, topic_id].toarray().T
        else:
            print(f"Unknown dtype '{type(topic_id)}'")

    def topic_distribution_for_document(self, doc_id=None, normalized=False):
        """Normalized: Divide each document's topic loadings by the sum of its topic loadings.
                       Results in the topic loadings for a document summing to 1.
        """
        if doc_id is None or (isinstance(doc_id, list) and (len(doc_id) == 0)):
            # all documents
            if normalized:
                return np.array(self.document_topic_matrix / self.document_topic_matrix.sum(axis=1))
            else:
                return self.document_topic_matrix.toarray()
        elif isinstance(doc_id, int) or (isinstance(doc_id, list) and (len(doc_id) == 1)):
            # one document
            if normalized:
                return np.array(self.document_topic_matrix[doc_id, :] / self.document_topic_matrix[doc_id, :].sum(axis=1))[0]
            else:
                return self.document_topic_matrix[doc_id, :].toarray()[0]
        elif isinstance(doc_id, list):
            # list of documents
            if normalized:
                return np.array(self.document_topic_matrix[doc_id, :] / self.document_topic_matrix[doc_id, :].sum(axis=1))
            else:
                return self.document_topic_matrix[doc_id, :].toarray()
        else:
            print(f"Unknown dtype '{type(doc_id)}'")

    def topic_distribution_for_word(self, word_id=None, normalized=False):
        """Normalized: Divide each word's topic loadings by the sum of its topic loadings.
                       Results in the topic loadings for a word summing to 1.
        """
        if word_id is None or (isinstance(word_id, list) and (len(word_id) == 0)):
            # all words
            if normalized:
                return np.array(self.topic_word_matrix / self.topic_word_matrix.sum(axis=1))
            else:
                return self.topic_word_matrix.toarray()
        elif isinstance(word_id, int) or (isinstance(word_id, list) and (len(word_id) == 1)):
            # one word
            if normalized:
                return np.array(
                    self.topic_word_matrix[:, word_id] / self.topic_word_matrix[:, word_id].sum(axis=0)).T[0]
            else:
                return self.topic_word_matrix[:, word_id].toarray().T[0]
        elif isinstance(word_id, list):
            # list of words
            if normalized:
                return np.array(
                    self.topic_word_matrix[:, word_id] / self.topic_word_matrix[:, word_id].sum(axis=0)).T
            else:
                return self.topic_word_matrix[:, word_id].toarray().T
        else:
            print(f"Unknown dtype '{type(word_id)}'")

    def topic_distribution_for_author(self, author_name: str, normalized=False):
        return self.topic_distribution_for_document(
            self.corpus.documents_by_author(author_name), normalized=normalized).mean(axis=0)

    def most_likely_topic_for_document(self, doc_id=None):
        if doc_id is None or (isinstance(doc_id, list) and (len(doc_id) == 0)):
            # all documents
            return np.argmax(self.document_topic_matrix.toarray(), axis=1)
        elif isinstance(doc_id, int) or (isinstance(doc_id, list) and (len(doc_id) == 1)):
            # one document
            return np.argmax(self.topic_distribution_for_document(doc_id), axis=0)
        elif isinstance(doc_id, list):
            # list of documents
            return np.argmax(self.topic_distribution_for_document(doc_id), axis=1)
        else:
            print(f"Unknown dtype '{type(doc_id)}'")

    def topic_frequency(self, topic: int, year: int = None, count: bool = False):
        """For a given topic, returns the percent out of all documents (or count of documents)
        for which it is the most likely topic, optionally slicing by year.
        """
        return self.topics_frequency(year=year, count=count)[topic]

    def topics_frequency(self, year: int = None, count: bool = False):
        """For each topic, returns the percent out of all documents (or count of documents)
        for which it is the most likely topic, optionally slicing by year.
        """
        if year is None:
            doc_ids = self.corpus.data_frame.index.tolist()
        else:
            doc_ids = self.corpus.doc_ids_year(year)

        # TODO make this work without the if statement
        if len(doc_ids) > 1:
            topic_count = Counter(self.most_likely_topic_for_document(doc_ids))
        else:
            topic_count = Counter([self.most_likely_topic_for_document(doc_ids)])
        frequency = np.array([topic_count[i] if i in topic_count else 0 for i in range(self.nb_topics)])

        if not count:
            frequency = frequency / len(doc_ids)

        return frequency

    def documents_for_topic(self, topic_id: int):
        """For a given topic, returns the document ids for which it is the most likely topic.
        Returned data structure is a list.
        """
        return self.corpus.data_frame.index[self.most_likely_topic_for_document() == topic_id].tolist()

    def documents_per_topic(self):
        """For each topic, returns the document ids for which each is the most likely topic.
        Returned data structure is a dictionary of lists, indexed by topic.
        """
        return {i: self.documents_for_topic(i) for i in range(self.nb_topics)}

    def affiliation_count(self, topic_id: int):
        """For a given topic, returns the count of affiliations for the documents
        for which it is the most likely topic.
        Returned data structure is a list of tuples sorted by affiliation count..
        """
        counts = {}
        doc_ids = self.documents_for_topic(topic_id)
        for i in doc_ids:
            # get the unique affiliation(s) for this document
            affiliations = set(self.corpus.affiliation(i))
            for affiliation in affiliations:
                if affiliation in counts:
                    counts[affiliation] += 1
                else:
                    counts[affiliation] = 1
        tuples = [(affiliation, count) for affiliation, count in counts.items()]
        tuples.sort(key=lambda x: x[1], reverse=True)
        return tuples

    def topic_distribution_for_new_document(self, text, normalized=False):
        """Predict the topic loading of a new document.
        """
        doc_topic_distr = self.model.transform(
            self.corpus.vectorizer.transform([text]))[0]
        if normalized:
            denom = doc_topic_distr.sum()
            doc_topic_distr = np.divide(doc_topic_distr, denom, out=np.zeros_like(doc_topic_distr), where=denom != 0)
        return doc_topic_distr

    def similar_documents(self, exemplar_vector, num_docs: int):
        """
        Given an exemplar topic loading vector, find similar documents.

        exemplar_vector can be a list or a 1-D numpy array

        returns a tuple of document ids and similarity scores
        """
        # Ensure it's 1-D and has one value per topic
        assert len(exemplar_vector) == self.nb_topics
        # normalize needs it to be 2-D
        exemplar_vector = np.array(exemplar_vector)[np.newaxis, :]
        # Compute pairwise similarities
        similarities = np.dot(csr_matrix(normalize(exemplar_vector, axis=1)),
                              normalize(self.document_topic_matrix, axis=1).T).toarray()[0]
        # Combine document indices and similarities
        similarities = [s for s in list(zip(self.corpus.data_frame.index.tolist(), similarities))]
        # Sort by most similar first
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]


class LatentDirichletAllocation(TopicModel):
    def __init__(self, corpus):
        self.trained = False
        self.model_type = 'LDA'
        self.corpus = corpus

    def infer_topics(
        self,
        num_topics: int = None,
        lda_algorithm: str = None,
        lda_alpha: float = None,
        lda_eta: float = None,
        lda_learning_method: str = None,
        lda_n_jobs: int = None,
        lda_n_iter: int = None,
        random_state=None,
        **kwargs,
    ):
        self.trained = True
        self.nb_topics = num_topics or 10
        self.lda_algorithm = lda_algorithm or 'variational'  # default sklearn
        self.lda_alpha = lda_alpha or 1 / num_topics
        self.lda_eta = lda_eta or 1 / num_topics
        self.lda_learning_method = lda_learning_method or 'batch'
        self.lda_n_jobs = lda_n_jobs or -1  # only for sklearn
        self.lda_n_iter = lda_n_iter or 2000  # only for lda library
        self.random_state = random_state

        if self.lda_algorithm == 'variational':
            # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
            self.model = LDA(
                n_components=num_topics,
                doc_topic_prior=self.lda_alpha,
                topic_word_prior=self.lda_eta,
                learning_method=self.lda_learning_method,
                n_jobs=self.lda_n_jobs,
                random_state=self.random_state,
            )
        elif self.lda_algorithm == 'gibbs':
            # https://github.com/ariddell/lda/
            self.model = lda.LDA(
                n_topics=num_topics,
                lda_n_iter=self.lda_n_iter,
                alpha=self.lda_alpha,
                eta=self.lda_eta,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"lda_algorithm must be either 'variational' or 'gibbs', got {self.lda_algorithm}")
        topic_document = self.model.fit_transform(self.corpus.sklearn_vector_space)

        self.topic_word_matrix = []
        self.document_topic_matrix = []
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(self.model.components_):
            for i in range(self.corpus.vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, self.corpus.vocabulary_size)).tocsr()
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
    def __init__(self, corpus):
        self.trained = False
        self.model_type = 'NMF'
        self.corpus = corpus

    def infer_topics(
        self,
        num_topics: int = None,
        nmf_init: str = None,
        nmf_solver: str = None,
        nmf_beta_loss: str = None,
        nmf_max_iter: int = None,
        nmf_alpha: float = None,
        nmf_l1_ratio: float = None,
        nmf_shuffle: bool = None,
        random_state=None,
        **kwargs,
    ):
        self.trained = True
        self.nb_topics = num_topics or 10
        self.nmf_init = nmf_init
        self.nmf_solver = nmf_solver or 'cd'
        self.nmf_beta_loss = nmf_beta_loss or 'frobenius'  # Used in 'mu' nmf_solver
        self.nmf_max_iter = nmf_max_iter or 200
        self.nmf_alpha = nmf_alpha or 0.0  # 0 = no regularization; used in the 'cd' nmf_solver
        self.nmf_l1_ratio = nmf_l1_ratio or 0.0  # 0 = L2, 1 = L1; used in the 'cd' nmf_solver
        self.nmf_shuffle = nmf_shuffle or False  # randomize the order of coordinates in the 'cd' nmf_solver
        self.random_state = random_state

        if (nmf_solver == 'mu') and (nmf_beta_loss not in ['frobenius', 'kullback-leibler', 'itakura-saito']):
            raise ValueError(f"nmf_beta_loss must be 'frobenius', 'kullback-leibler', or 'itakura-saito', got {nmf_beta_loss}")

        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        self.model = NMF(
            n_components=num_topics,
            init=self.nmf_init,
            solver=self.nmf_solver,
            beta_loss=self.nmf_beta_loss,
            max_iter=self.nmf_max_iter,
            alpha=self.nmf_alpha,
            l1_ratio=self.nmf_l1_ratio,
            shuffle=self.nmf_shuffle,
            random_state=self.random_state,
        )
        topic_document = self.model.fit_transform(self.corpus.sklearn_vector_space)

        self.topic_word_matrix = []
        self.document_topic_matrix = []
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(self.model.components_):
            for i in range(self.corpus.vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, self.corpus.vocabulary_size)).tocsr()
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
