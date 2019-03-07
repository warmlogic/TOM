# coding: utf-8
import configparser
from pathlib import Path
import os
import numpy as np
import tom_lib.utils as ut
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization
# import nltk
import logging
logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"

# # Download stopwords from NLTK
# nltk.download('stopwords')

config_filepath = 'config.ini'
config = configparser.ConfigParser(allow_no_value=True)
try:
    config.read(config_filepath)
except OSError as e:
    logger.error(f'Config file {config_filepath} not found. Did you set it up?')

# Read parameters
webserver_section = 'webserver'
data_dir = config[webserver_section].get('data_dir', '', vars=os.environ)
if not data_dir:
    data_dir = '.'
data_dir = Path(data_dir)
docs_filename = config[webserver_section].get('docs_filename', '')
if not docs_filename:
    raise ValueError(f'docs_filename not specified in {config_filepath}')

source_filepath = data_dir / docs_filename
# Ensure data exists
if not source_filepath.exists():
    raise OSError(f'Documents file does not exist: {source_filepath}')

language = config[webserver_section].get('language', None)
assert (isinstance(language, str) and language in ['english']) or (isinstance(language, list)) or (language is None)
# ignore words which relative frequency is > than max_relative_frequency
max_relative_frequency = config[webserver_section].getfloat('max_relative_frequency', 0.8)
# ignore words which absolute frequency is < than min_absolute_frequency
min_absolute_frequency = config[webserver_section].getint('min_absolute_frequency', 5)
num_topics = config[webserver_section].getint('num_topics', 15)
# 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
vectorization = config[webserver_section].get('vectorization', 'tfidf')
n_gram = config[webserver_section].getint('n_gram', 1)
max_features = config[webserver_section].get('max_features', None)
if isinstance(max_features, str):
    if max_features.isnumeric():
        max_features = int(max_features)
    elif max_features == 'None':
        max_features = None
assert isinstance(max_features, int) or (max_features is None)
sample = config[webserver_section].getfloat('sample', 1.0)
top_words_description = config[webserver_section].getint('top_words_description', 5)
top_words_cloud = config[webserver_section].getint('top_words_cloud', 5)
model_type = config[webserver_section].get('model_type', 'NMF')
nmf_beta_loss = config[webserver_section].get('nmf_beta_loss', 'frobenius')
lda_algorithm = config[webserver_section].get('lda_algorithm', 'variational')

# read assessment config parameters
assess_section = 'assess'
min_num_topics = config[assess_section].getint('min_num_topics', 11)
max_num_topics = config[assess_section].getint('max_num_topics', 49)
step = config[assess_section].getint('step', 2)
greene_tao = config[assess_section].getint('greene_tao', 10)
greene_top_n_words = config[assess_section].getint('greene_top_n_words', 10)
greene_sample = config[assess_section].getfloat('greene_sample', 0.8)
iterations = config[assess_section].getint('iterations', 10)
# perplexity_train_size = config[assess_section].getfloat('perplexity_train_size', 0.7)
verbose = config[assess_section].getboolean('verbose', )

if model_type not in ['NMF', 'LDA']:
    raise ValueError('model_type must be NMF or LDA')

if model_type == 'NMF':
    if nmf_beta_loss not in ['frobenius', 'kullback-leibler', 'itakura-saito']:
        raise ValueError(f"For NMF, 'beta_loss' must be 'frobenius', 'kullback-leibler', or 'itakura-saito', got '{nmf_beta_loss}'")
    if vectorization == 'tf':
        raise ValueError(f"for NMF, 'vectorization' should be 'tfidf', got '{vectorization}'")
elif model_type == 'LDA':
    if lda_algorithm not in ['variational', 'gibbs']:
        raise ValueError(f"For LDA, 'lda_algorithm' must be 'variational' or 'gibbs', got '{lda_algorithm}'")
    if vectorization == 'tfidf':
        raise ValueError(f"for LDA, 'vectorization' should be 'tf', got '{vectorization}'")

# Load and prepare a corpus
logger.info(f'Loading documents: {source_filepath}')
corpus = Corpus(source_filepath=source_filepath,
                language=language,
                vectorization=vectorization,
                n_gram=n_gram,
                max_relative_frequency=max_relative_frequency,
                min_absolute_frequency=min_absolute_frequency,
                max_features=max_features,
                sample=sample,
                full_text_col='orig_text',
                )
logger.info(f'Corpus size: {corpus.size:,}')
logger.info(f'Vocabulary size: {len(corpus.vocabulary):,}')

# Initialize topic model
if model_type == 'NMF':
    topic_model = NonNegativeMatrixFactorization(corpus=corpus)
elif model_type == 'LDA':
    topic_model = LatentDirichletAllocation(corpus=corpus)

# Estimate the optimal number of topics
logger.info('Estimating the number of topics to choose. This could take a while...')
viz = Visualization(topic_model)
logger.info(f'Total number of topics to assess: {len(np.arange(min_num_topics, max_num_topics + 1, step))}')
logger.info(f'Topic numbers: {np.arange(min_num_topics, max_num_topics + 1, step)}')

viz.plot_greene_metric(
    min_num_topics=min_num_topics,
    max_num_topics=max_num_topics,
    step=step,
    tao=greene_tao,
    top_n_words=greene_top_n_words,
    sample=greene_sample,
    verbose=verbose,
)

viz.plot_arun_metric(
    min_num_topics=min_num_topics,
    max_num_topics=max_num_topics,
    step=step,
    iterations=iterations,
    verbose=verbose,
)

viz.plot_brunet_metric(
    min_num_topics=min_num_topics,
    max_num_topics=max_num_topics,
    step=step,
    iterations=iterations,
    verbose=verbose,
)

# viz.plot_perplexity_metric(
#     min_num_topics=min_num_topics,
#     max_num_topics=max_num_topics,
#     step=step,
#     train_size=perplexity_train_size,
#     verbose=verbose,
# )

# Infer topics
logger.info('Inferring topics')
topic_model.infer_topics(num_topics=num_topics)

# Save model on disk
topic_model_filepath = data_dir / f'{model_type}_{source_filepath.stem}_{num_topics}topics.pickle'
logger.info(f'Saving topic model: {topic_model_filepath}')
ut.save_topic_model(topic_model, topic_model_filepath)

# # Load model from disk:
# logger.info(f'Loading topic model: {topic_model_filepath}')
# topic_model = ut.load_topic_model(topic_model_filepath)

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
