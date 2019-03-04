# coding: utf-8
import tom_lib.utils as ut
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization
import nltk

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"

# Download stopwords from NLTK
nltk.download('stopwords')

# Load and prepare a corpus
print('Load documents from CSV')
# ignore words which relative frequency is > than max_relative_frequency
max_relative_frequency = 0.8
# ignore words which absolute frequency is < than min_absolute_frequency
min_absolute_frequency = 5
# 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
vectorization = 'tfidf'
source_filepath = 'input/egc_lemmatized.csv'
# language for stop words
language = 'french'

corpus = Corpus(source_filepath=source_filepath,
                language=language,
                vectorization=vectorization,
                max_relative_frequency=max_relative_frequency,
                min_absolute_frequency=min_absolute_frequency,
                )
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))

# Instantiate a topic model
topic_model = NonNegativeMatrixFactorization(corpus)

# # Estimate the optimal number of topics
# print('Estimating the number of topics...')
# viz = Visualization(topic_model)
# min_num_topics = 11
# max_num_topics = 21
# step = 2
# tao = 10
# top_n_words = 10
# iterations = 10
# # train_size = 0.7
# verbose = True
# viz.plot_greene_metric(
#     min_num_topics=min_num_topics,
#     max_num_topics=max_num_topics,
#     step=step,
#     tao=tao,
#     top_n_words=top_n_words,
#     verbose=verbose,
# )
# viz.plot_arun_metric(
#     min_num_topics=min_num_topics,
#     max_num_topics=max_num_topics,
#     step=step,
#     iterations=iterations,
#     verbose=verbose,
# )
# viz.plot_brunet_metric(
#     min_num_topics=min_num_topics,
#     max_num_topics=max_num_topics,
#     step=step,
#     iterations=iterations,
#     verbose=verbose,
# )
# # viz.plot_perplexity_metric(
# #     min_num_topics=min_num_topics,
# #     max_num_topics=max_num_topics,
# #     step=step,
# #     train_size=train_size,
# #     verbose=verbose,
# # )

# Infer topics
print('Inferring topics...')
num_topics = 15
topic_model.infer_topics(num_topics=num_topics)
# Save model on disk
ut.save_topic_model(topic_model, 'NMF_EGC_15topics.pickle')
# Load model from disk: topic_model = ut.load_topic_model('NMF_EGC_15topics.pickle')

# Print results
print('\nTopics:')
topic_model.print_topics(num_words=10)
print('\nTopic distribution for document 0:',
      topic_model.topic_distribution_for_document(0))
print('\nMost likely topic for document 0:',
      topic_model.most_likely_topic_for_document(0))
print('\nFrequency of topics:',
      topic_model.topics_frequency())
print('\nTop 10 most relevant words for topic 2:',
      topic_model.top_words(2, 10))
