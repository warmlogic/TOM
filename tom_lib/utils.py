# coding: utf-8
import codecs
import json
import pickle
import numpy as np
from typing import List

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


def print_matrix(matrix):
    n_r = len(matrix[:, 0])
    for i in range(n_r):
        print(matrix[i, :])


def save_topic_model(topic_model, file_path):
    pickle.dump(topic_model, open(file_path, 'wb'))


def load_topic_model(file_path):
    return pickle.load(open(file_path, 'rb'))


def save_word_distribution(distribution, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('word\tweight\n')
        for weighted_word in distribution:
            f.write(f'{weighted_word[0]}\t{weighted_word[1]}\n')


def save_topic_distribution(distribution, file_path, topic_description: List=None):
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
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('date\tfrequency\n')
        for date, frequency in evolution:
            f.write(f'{date}\t{frequency}\n')


def save_affiliation_repartition(affiliation_repartition, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('affiliation\tcount\n')
        for (affiliation, count) in affiliation_repartition:
            f.write(f'{affiliation}\t{count}\n')


def save_topic_number_metrics_data(path, range_, data, step=None, metric_type=''):
    with open(path, "w") as filo:
        filo.write(f"k\t{metric_type}_value\n")
        for idx, range_i in enumerate(np.arange(range_[0], range_[1] + 1, step)):
            filo.write(f'{range_i}\t{data[idx]}\n')


def save_topic_cloud(topic_model, file_path, top_words=5):
    json_graph = {}
    json_nodes = []
    json_links = []
    for i in range(topic_model.nb_topics):
        description = []
        for weighted_word in topic_model.top_words(i, top_words):
            description.append(weighted_word[0])
        json_nodes.append({'name': i,
                           'frequency': topic_model.topic_frequency(i),
                           'description': f"Topic {i}: {', '.join(description)}",
                           'group': i})
    json_graph['nodes'] = json_nodes
    json_graph['links'] = json_links
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_graph, fp, indent=4, separators=(',', ': '))


def save_json_object(json_object, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_object, fp, indent=4, separators=(',', ': '))
