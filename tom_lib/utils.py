from typing import List
import argparse
import configparser
import logging
import codecs
import json
import pickle
import pandas as pd
import numpy as np

logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)


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
    except OSError:
        logger.exception(f'Config file {config_filepath} not found. Did you set it up?')
    return config


def print_matrix(matrix):
    n_r = len(matrix[:, 0])
    for i in range(n_r):
        print(matrix[i, :])


def save_topic_model(topic_model, file_path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(topic_model, open(file_path, 'wb'))


def load_topic_model(file_path):
    return pickle.load(open(file_path, 'rb'))


def save_word_distribution(distribution, file_path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('word\tweight\n')
        for weighted_word in distribution:
            f.write(f'{weighted_word[0]}\t{weighted_word[1]}\n')


def save_topic_distribution(distribution, file_path, topic_description: List = None):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if topic_description:
        if len(topic_description) != len(distribution):
            topic_description = None
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('topic\tweight\n')
        for i in range(len(distribution)):
            if topic_description:
                f.write(f'{topic_description[i]}\t{distribution[i]}\n')
            else:
                f.write(f'Topic {i}\t{distribution[i]}\n')


def save_topic_evolution(evolution, file_path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('date\tfrequency\n')
        for date, frequency in evolution:
            f.write(f'{date}\t{frequency}\n')


def save_affiliation_count(affiliation_count, file_path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('affiliation\tcount\n')
        for (affiliation, count) in affiliation_count:
            f.write(f'{affiliation}\t{count}\n')


def save_topic_number_metrics_data(file_path, range_, data, step=None, metric_type=''):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as filo:
        filo.write(f"k\t{metric_type}_value\n")
        for idx, range_i in enumerate(np.arange(range_[0], range_[1] + 1, step)):
            filo.write(f'{range_i}\t{data[idx]}\n')


def save_topic_cloud(topic_model, file_path, top_words=5):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    json_graph = {}
    json_nodes = []
    json_links = []
    for i in range(topic_model.nb_topics):
        description = [weighted_word[0] for weighted_word in topic_model.top_words(i, top_words)]
        json_nodes.append({'name': i,
                           'frequency': topic_model.topic_frequency(i),
                           'description': f"Topic {i}: {', '.join(description)}",
                           'group': i})
    json_graph['nodes'] = json_nodes
    json_graph['links'] = json_links
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_graph, fp, indent=4, separators=(',', ': '))


def save_json_object(json_object, file_path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_object, fp, indent=4, separators=(',', ': '))


def save_top_words(num_top_words_save, topic_model, file_path):
    df_top_words = pd.DataFrame(
        data=np.array(topic_model.top_words_topics(num_words=num_top_words_save)).T,
        index=range(1, num_top_words_save + 1),
        columns=[f'Topic {i}' for i in range(topic_model.nb_topics)],
    )
    df_top_words.index.name = 'word rank'
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    df_top_words.to_csv(f'{file_path}.csv')
    with pd.ExcelWriter(f'{file_path}.xlsx') as writer:
        df_top_words.to_excel(writer, index=True)


def delete_folder(pth):
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
