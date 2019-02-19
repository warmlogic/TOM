# coding: utf-8"
import matplotlib as mpl

from tom_lib.utils import save_topic_number_metrics_data

mpl.use("Agg")  # To be able to create figures on a headless server (no DISPLAY variable)
import matplotlib.pyplot as plt
import codecs
import numpy as np
import json
from pathlib import Path

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"


class Visualization:

    def __init__(self, topic_model, output_dir=None):
        self.topic_model = topic_model
        if output_dir is None:
            self.output_dir = Path('output')
        else:
            if isinstance(output_dir, str):
                self.output_dir = Path(output_dir)
            elif isinstance(output_dir, Path):
                self.output_dir = output_dir
            else:
                raise TypeError(f"'output_dir' of type {type(output_dir)} not a valid type")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_topic_distribution(self, doc_id, file_name='topic_distribution.png'):
        file_path = self.output_dir / file_name
        distribution = self.topic_model.topic_distribution_for_document(doc_id)
        data_x = range(0, len(distribution))
        plt.clf()
        plt.xticks(np.arange(0, len(distribution), 1.0))
        plt.bar(data_x, distribution, align='center')
        plt.title('Topic distribution')
        plt.ylabel('probability')
        plt.xlabel('topic')
        plt.savefig(file_path)

    def plot_word_distribution(self, topic_id, nb_words=10, file_name='word_distribution.png'):
        file_path = self.output_dir / file_name
        data_x = []
        data_y = []
        distribution = self.topic_model.top_words(topic_id, nb_words)
        for weighted_word in distribution:
            data_x.append(weighted_word[0])
            data_y.append(weighted_word[1])
        plt.clf()
        plt.bar(range(len(data_x)), data_y, align='center')
        plt.xticks(range(len(data_x)), data_x, size='small', rotation='vertical')
        plt.title('Word distribution')
        plt.ylabel('probability')
        plt.xlabel('word')
        plt.savefig(file_path)

    def plot_greene_metric(self, min_num_topics=10, max_num_topics=20, tao=10, step=5, top_n_words=10, verbose=True):
        greene_stability = self.topic_model.greene_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            top_n_words=top_n_words,
            tao=tao,
            verbose=verbose,
        )
        plt.clf()
        plt.plot(np.arange(min_num_topics, max_num_topics + 1, step), greene_stability, 'o-')
        plt.title('Greene et al. metric')
        plt.xlabel('Number of topics')
        plt.ylabel('Stability')
        file_path_fig = self.output_dir / 'greene.png'
        file_path_data = self.output_dir / 'greene.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=greene_stability, step=step, metric_type='greene')

    def plot_arun_metric(self, min_num_topics=10, max_num_topics=50, step=5, iterations=10, verbose=True):
        symmetric_kl_divergence = self.topic_model.arun_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            verbose=verbose,
        )
        plt.clf()
        plt.plot(range(min_num_topics, max_num_topics + 1, step), symmetric_kl_divergence, 'o-')
        plt.title('Arun et al. metric')
        plt.xlabel('Number of topics')
        plt.ylabel('Symmetric KL Divergence')
        file_path_fig = self.output_dir / 'arun.png'
        file_path_data = self.output_dir / 'arun.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=symmetric_kl_divergence, step=step, metric_type='arun')

    def plot_brunet_metric(self, min_num_topics=10, max_num_topics=50, step=5, iterations=10, verbose=True):
        cophenetic_correlation = self.topic_model.brunet_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            verbose=verbose,
        )
        plt.clf()
        plt.plot(range(min_num_topics, max_num_topics + 1, step), cophenetic_correlation, 'o-')
        plt.title('Brunet et al. metric')
        plt.xlabel('Number of topics')
        plt.ylabel('Cophenetic correlation coefficient')
        file_path_fig = self.output_dir / 'brunet.png'
        file_path_data = self.output_dir / 'brunet.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=cophenetic_correlation, step=step, metric_type='brunet')

    def plot_perplexity_metric(self, min_num_topics=10, max_num_topics=20, step=5, train_size=0.7, verbose=True):
        train_perplexities, test_perplexities = self.topic_model.perplexity_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            train_size=train_size,
            verbose=verbose,
        )
        if (len(train_perplexities) > 0) and (len(test_perplexities) > 0):
            plt.clf()
            plt.plot(np.arange(min_num_topics, max_num_topics + 1, step), train_perplexities, 'o-', label='Train')
            plt.plot(np.arange(min_num_topics, max_num_topics + 1, step), test_perplexities, 'o-', label='Test')
            plt.title('Perplexity metric')
            plt.xlabel('Number of topics')
            plt.ylabel('Perplexity')
            plt.legend(loc='best')
            file_path_fig = self.output_dir / 'perplexity.png'
            file_path_data_train = self.output_dir / 'perplexity_train.tsv'
            file_path_data_test = self.output_dir / 'perplexity_test.tsv'
            plt.savefig(file_path_fig)
            save_topic_number_metrics_data(
                file_path_data_train,
                range_=(min_num_topics, max_num_topics),
                data=train_perplexities, step=step, metric_type='perplexity')
            save_topic_number_metrics_data(
                file_path_data_test,
                range_=(min_num_topics, max_num_topics),
                data=test_perplexities, step=step, metric_type='perplexity')

    def topic_cloud(self, file_name='topic_cloud.json'):
        file_path = self.output_dir / file_name
        json_graph = {}
        json_nodes = []
        json_links = []
        for i in range(self.topic_model.nb_topics):
            description = []
            for weighted_word in self.topic_model.top_words(i, 5):
                description.append(weighted_word[1])
            json_nodes.append({'name': f'topic{i}',
                               'frequency': self.topic_model.topic_frequency(i),
                               'description': ', '.join(description),
                               'group': i})
        json_graph['nodes'] = json_nodes
        json_graph['links'] = json_links
        with codecs.open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(json_graph, fp, indent=4, separators=(',', ': '))
