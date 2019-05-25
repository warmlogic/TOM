# coding: utf-8
from typing import List, Tuple

import matplotlib as mpl

from tom_lib.utils import save_topic_number_metrics_data

mpl.use("Agg")  # To be able to create figures on a headless server (no DISPLAY variable)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import codecs
import numpy as np
import json
from pathlib import Path


class Visualization:

    def __init__(self, topic_model, output_dir=None):
        self.topic_model = topic_model
        if output_dir is None:
            self.output_dir = Path(f'output_{self.topic_model.model_type}_{self.topic_model.nb_topics}_topics')
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

    def plot_greene_metric(self, min_num_topics=10, max_num_topics=20, tao=10, step=5, top_n_words=10,
                           sample=0.8, beta_loss='frobenius', algorithm='variational', verbose=True):
        greene_stability = self.topic_model.greene_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            top_n_words=top_n_words,
            tao=tao,
            sample=sample,
            beta_loss=beta_loss,
            algorithm=algorithm,
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

    def plot_arun_metric(self, min_num_topics=10, max_num_topics=50, step=5, iterations=10,
                         beta_loss='frobenius', algorithm='variational', verbose=True):
        symmetric_kl_divergence = self.topic_model.arun_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            beta_loss=beta_loss,
            algorithm=algorithm,
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

    def plot_brunet_metric(self, min_num_topics=10, max_num_topics=50, step=5, iterations=10,
                           beta_loss='frobenius', algorithm='variational', verbose=True):
        cophenetic_correlation = self.topic_model.brunet_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            beta_loss=beta_loss,
            algorithm=algorithm,
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

    def plot_perplexity_metric(self, min_num_topics=10, max_num_topics=20, step=5, train_size=0.7,
                               algorithm='variational', verbose=True):
        train_perplexities, test_perplexities = self.topic_model.perplexity_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            train_size=train_size,
            algorithm=algorithm,
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
                               'description': f"Topic {i}: {', '.join(description)}",
                               'group': i})
        json_graph['nodes'] = json_nodes
        json_graph['links'] = json_links
        with codecs.open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(json_graph, fp, indent=4, separators=(',', ': '))

    def plot_docs_above_thresh(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        thresh: float = 0.5,
        kind: str = 'count',
        n_words: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
    ):
        '''plot the number of documents associated with each topic, above some threshold
        kind = 'count' or 'percent'
        '''

        fig, ax = plt.subplots(figsize=figsize)

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if kind == 'count':
            if normalized:
                result = ((
                    self.topic_model.document_topic_matrix / self.topic_model.document_topic_matrix.sum(axis=1)) > thresh).sum(axis=0)
            else:
                result = (self.topic_model.document_topic_matrix > thresh).sum(axis=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
            ax.set_ylabel('Count of documents')
        elif kind == 'percent':
            if normalized:
                result = ((
                    (self.topic_model.document_topic_matrix / self.topic_model.document_topic_matrix.sum(axis=1)) > thresh).sum(axis=0) / self.topic_model.corpus.size)
            else:
                result = ((
                    self.topic_model.document_topic_matrix > thresh).sum(axis=0) / self.topic_model.corpus.size)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            ax.set_ylabel('Percent of documents')

        result = np.array(result)[0]
        result = result[[tc in topic_cols for tc in topic_cols_all]]
        sns.barplot(x=topic_cols, y=result, ax=ax)
        # result = pd.DataFrame(data=result, columns=topic_cols_all)[topic_cols]
        # sns.barplot(ax=ax, data=result)

        title_str = f'Documents above {thresh} topic loading'
        if normalized:
            title_str = f'{title_str} (normalized)'
        title_str = f'{title_str}; {self.topic_model.corpus.size:,} total docs'

        ax.set_title(title_str)
        fig.autofmt_xdate()

        if savefig:
            filename_out = f'hist_above_thresh_{kind}_{len(topic_cols)}_topics.png'
            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=150, transparent=False)
            plt.close('all')
        else:
            plt.show()

        return fig, ax

    def plot_heatmap(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        mask_thresh: float = None,
        cmap=None,
        vmax: float = None,
        vmin: float = None,
        fmt: str = '.2f',
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
    ):
        '''Plot a heatmap of a correlation dataframe
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if normalized:
            corr = np.corrcoef(
                (self.topic_model.document_topic_matrix /
                    self.topic_model.document_topic_matrix.sum(axis=1)).T)
            norm_str = 'normalized'
        else:
            corr = np.corrcoef(self.topic_model.document_topic_matrix.todense().T)
            norm_str = ''

        corr = pd.DataFrame(data=corr, columns=topic_cols_all, index=topic_cols_all)
        corr = corr.loc[topic_cols, topic_cols]

        if mask_thresh is None:
            mask_thresh = 0
        if figsize is None:
            figsize = (max(25, len(topic_cols)), max(15, min(len(topic_cols) // 1.2, 15)))
        if cmap is None:
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if vmax is None:
            vmax = corr.max().max()
        # vmax=0.25
        # vmin=-vmax
        if vmin is None:
            vmin = corr.min().min()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(corr, ax=ax, center=0, annot=True, fmt=fmt,
                    vmin=vmin, vmax=vmax,
                    mask=((corr > -mask_thresh) & (corr < mask_thresh)),
                    cmap=cmap,
                    # square=True,
                    )

        ax.hlines(range(1, corr.shape[0]), *ax.get_xlim(), lw=0.5)
        ax.vlines(range(1, corr.shape[1]), *ax.get_ylim(), lw=0.5)

        fig.autofmt_xdate()
        fig.tight_layout()

        if savefig:
            filename_out = f'topic-topic_corr{norm_str}_{len(topic_cols)}_topics.png'
            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=150, transparent=False)
            plt.close('all')
        else:
            plt.show()

        return fig, ax

    def plot_clustermap(
        self,
        normalized: bool = True,
        mask_thresh: float = None,
        cmap=None,
        vmax: float = None,
        vmin: float = None,
        fmt: str = '.2f',
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        metric: str = None,
        method: str = None,
    ):
        '''Plot a clustermap of a correlation dataframe (df.corr())
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))

        if normalized:
            corr = np.corrcoef(
                (self.topic_model.document_topic_matrix /
                    self.topic_model.document_topic_matrix.sum(axis=1)).T)
            norm_str = 'normalized'
        else:
            corr = np.corrcoef(self.topic_model.document_topic_matrix.todense().T)
            norm_str = ''

        corr = pd.DataFrame(data=corr, columns=topic_cols_all, index=topic_cols_all)

        if mask_thresh is None:
            mask_thresh = 0
        if figsize is None:
            figsize = (max(25, len(topic_cols_all)), max(15, min(len(topic_cols_all) // 1.2, 15)))
        if cmap is None:
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if vmax is None:
            vmax = corr.max().max()
        # vmax=0.25
        # vmin=-vmax
        if vmin is None:
            vmin = corr.min().min()

        if metric is None:
            metric = 'euclidean'
            # metric = 'correlation'

        if method is None:
            # method = 'complete'
            method = 'average'
            # method = 'ward'

        g = sns.clustermap(
            corr,
            center=0, annot=True, fmt=fmt,
            metric=metric,
            method=method,
            vmin=vmin, vmax=vmax,
            mask=((corr > -mask_thresh) & (corr < mask_thresh)),
            cmap=cmap,
            figsize=figsize,
        )

        g.ax_heatmap.hlines(range(1, corr.shape[0]), *g.ax_heatmap.get_xlim(), lw=0.5)
        g.ax_heatmap.vlines(range(1, corr.shape[1]), *g.ax_heatmap.get_ylim(), lw=0.5)

        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=30, ha='right')

        if savefig:
            filename_out = f'topic-topic_corr_grouped{norm_str}_{len(topic_cols_all)}_topics.png'
            # save image to disk
            g.savefig(self.output_dir / '{}'.format(filename_out), dpi=150, transparent=False)
            # save values to csv
            corr.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind].to_csv('{}.csv'.format(filename_out))
        else:
            plt.show()

        return g
