# coding: utf-8
import argparse
import configparser
from datetime import datetime
from pathlib import Path
import os
# import tom_lib.utils as ut
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization
# import nltk
import logging

logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)

# # Download stopwords from NLTK
# nltk.download('stopwords')


def get_parser():
    """
    Creates a new argument parser
    """
    parser = argparse.ArgumentParser(prog='assess_topics')
    parser.add_argument('--config_filepath', '-c', type=str, nargs='?', default='config.ini')
    return parser


def get_config(config_filepath):
    """
    Read the specified config file
    """
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config.read(config_filepath)
    except OSError as e:
        logger.error(f'Config file {config_filepath} not found. Did you set it up?')
    return config


def main(config_infer):
    # get the current datetime string for use in the output directory name
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    data_dir = config_infer.get('data_dir', '', vars=os.environ)
    if not data_dir:
        data_dir = '.'
    data_dir = Path(data_dir)
    docs_filename = config_infer.get('docs_filename', '')
    if not docs_filename:
        raise ValueError(f'docs_filename not specified in {config_filepath}')

    source_filepath = data_dir / docs_filename
    # Ensure data exists
    if not source_filepath.exists():
        raise OSError(f'Documents file does not exist: {source_filepath}')

    # read data config parameters
    language = config_infer.get('language', None)
    assert (isinstance(language, str) and language in ['english']) or (isinstance(language, list)) or (language is None)
    model_type = config_infer.get('model_type', 'NMF')
    random_state = config_infer.getint('random_state', None)
    nmf_beta_loss = config_infer.get('nmf_beta_loss', 'frobenius')
    lda_algorithm = config_infer.get('lda_algorithm', 'variational')
    # ignore words which relative frequency is > than max_relative_frequency
    max_relative_frequency = config_infer.getfloat('max_relative_frequency', 0.8)
    # ignore words which absolute frequency is < than min_absolute_frequency
    min_absolute_frequency = config_infer.getint('min_absolute_frequency', 5)
    # 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
    vectorization = config_infer.get('vectorization', 'tfidf')
    n_gram = config_infer.getint('n_gram', 1)
    max_features = config_infer.get('max_features', None)
    if isinstance(max_features, str):
        if max_features.isnumeric():
            max_features = int(max_features)
        elif max_features == 'None':
            max_features = None
    assert isinstance(max_features, int) or (max_features is None)
    sample = config_infer.getfloat('sample', 1.0)

    # read assessment config parameters
    min_num_topics = config_infer.getint('min_num_topics', 11)
    max_num_topics = config_infer.getint('max_num_topics', 49)
    step = config_infer.getint('step', 2)
    greene_tao = config_infer.getint('greene_tao', 10)
    greene_top_n_words = config_infer.getint('greene_top_n_words', 10)
    greene_sample = config_infer.getfloat('greene_sample', 0.8)
    arun_iterations = config_infer.getint('arun_iterations', 10)
    brunet_iterations = config_infer.getint('brunet_iterations', 10)
    coherence_w2v_top_n_words = config_infer.getint('coherence_w2v_top_n_words', 10)
    coherence_w2v_size = config_infer.getint('coherence_w2v_size', 100)
    # perplexity_train_size = config_infer.getfloat('perplexity_train_size', 0.7)
    verbose = config_infer.getboolean('verbose', )

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
    corpus = Corpus(
        source_filepath=source_filepath,
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
    num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
    logger.info(f'Total number of topics to infer: {len(num_topics_infer)}')
    logger.info(f'Topic numbers: {list(num_topics_infer)}')

    output_dir = f'assess_{topic_model.model_type}_{source_filepath.stem}_{now_str}'

    viz = Visualization(topic_model, output_dir=output_dir)

    logger.info('Estimating the number of topics to choose. This could take a while...')
    logger.info(f'Will save results to: {viz.output_dir}')

    logger.info('Assessing Greene metric')
    viz.plot_greene_metric(
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        step=step,
        tao=greene_tao,
        top_n_words=greene_top_n_words,
        sample=greene_sample,
        random_state=random_state,
        verbose=verbose,
    )

    logger.info('Assessing Arun metric')
    viz.plot_arun_metric(
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        step=step,
        iterations=arun_iterations,
        random_state=random_state,
        verbose=verbose,
    )

    logger.info('Assessing Coherence Word2Vec metric')
    viz.plot_coherence_w2v_metric(
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        step=step,
        top_n_words=coherence_w2v_top_n_words,
        w2v_size=coherence_w2v_size,
        random_state=random_state,
        verbose=verbose,
    )

    logger.info('Assessing Brunet metric')
    viz.plot_brunet_metric(
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        step=step,
        iterations=brunet_iterations,
        random_state=random_state,
        verbose=verbose,
    )

    # logger.info('Assessing perplexity metric')
    # viz.plot_perplexity_metric(
    #     min_num_topics=min_num_topics,
    #     max_num_topics=max_num_topics,
    #     step=step,
    #     train_size=perplexity_train_size,
    #     random_state=random_state,
    #     verbose=verbose,
    # )


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config_filepath = args.config_filepath
    config = get_config(config_filepath)

    config_section = 'infer'

    main(config[config_section])
