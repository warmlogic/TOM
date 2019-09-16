# coding: utf-8
from typing import List, Tuple
import codecs
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px

from tom_lib.utils import save_topic_number_metrics_data

sns.set(rc={"lines.linewidth": 2})
sns.set_style("whitegrid")

mpl.use("Agg")  # To be able to create figures on a headless server (no DISPLAY variable)


def split_string_sep(string: str, sep: str = None):
    """Split a string on spaces and put together with newlines
    """
    if sep is None:
        sep = ' '
    string_new = sep.join(string.split(sep=sep)[:2])
    if len(string.split()) > 2:
        string_new = '\n'.join([string_new, sep.join(string.split()[2:4])])
        if len(string.split()) > 4:
            string_new = '\n'.join([string_new, sep.join(string.split()[4:7])])
            if len(string.split()) > 7:
                string_new = '\n'.join([string_new, sep.join(string.split()[7:])])
    return string_new


def split_string_nchar(string: str, nchar: int = None):
    """Split a string into a given number of chunks based on number of characters
    """
    if nchar is None:
        nchar = 25
    return '\n'.join([string[(i * nchar):(i + 1) * nchar] for i in range(int(np.ceil(len(string) / nchar)))])


class Visualization:
    def __init__(self, topic_model, output_dir=None):
        self.topic_model = topic_model

        if output_dir is None:
            if self.topic_model.trained:
                self.output_dir = Path(f'output_{self.topic_model.model_type}_{self.topic_model.nb_topics}_topics')
            else:
                self.output_dir = Path(f'output_{self.topic_model.model_type}')
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

    def plot_greene_metric(
        self,
        min_num_topics=10,
        max_num_topics=20,
        tao=10,
        step=5,
        top_n_words=10,
        sample=0.8,
        beta_loss='frobenius',
        algorithm='variational',
        random_state=None,
        verbose=True,
    ):
        greene_stability = self.topic_model.greene_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            top_n_words=top_n_words,
            tao=tao,
            sample=sample,
            beta_loss=beta_loss,
            algorithm=algorithm,
            random_state=random_state,
            verbose=verbose,
        )
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        plt.clf()
        plt.plot(num_topics_infer, greene_stability, 'o-')
        plt.xticks(num_topics_infer)
        plt.title('Greene et al. metric (higher is better)')
        plt.xlabel('Number of Topics')
        plt.ylabel('Stability')
        # find and annotate the maximum point on the plot
        ymax = max(greene_stability)
        xpos = greene_stability.index(ymax)
        best_k = num_topics_infer[xpos]
        plt.annotate(f'k={best_k}', xy=(best_k, ymax), xytext=(best_k, ymax), textcoords='offset points', fontsize=16)
        file_path_fig = self.output_dir / 'greene.png'
        file_path_data = self.output_dir / 'greene.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=greene_stability, step=step, metric_type='greene')

    def plot_arun_metric(
        self,
        min_num_topics=10,
        max_num_topics=50,
        step=5,
        iterations=10,
        beta_loss='frobenius',
        algorithm='variational',
        random_state=None,
        verbose=True,
    ):
        symmetric_kl_divergence = self.topic_model.arun_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            beta_loss=beta_loss,
            algorithm=algorithm,
            random_state=random_state,
            verbose=verbose,
        )
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        plt.clf()
        plt.plot(num_topics_infer, symmetric_kl_divergence, 'o-')
        plt.xticks(num_topics_infer)
        plt.title('Arun et al. metric (lower is better)')
        plt.xlabel('Number of Topics')
        plt.ylabel('Symmetric KL Divergence')
        # find and annotate the maximum point on the plot
        ymin = min(symmetric_kl_divergence)
        xpos = symmetric_kl_divergence.index(ymin)
        best_k = num_topics_infer[xpos]
        plt.annotate(f'k={best_k}', xy=(best_k, ymin), xytext=(best_k, ymin), textcoords='offset points', fontsize=16)
        file_path_fig = self.output_dir / 'arun.png'
        file_path_data = self.output_dir / 'arun.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=symmetric_kl_divergence, step=step, metric_type='arun')

    def plot_brunet_metric(
        self,
        min_num_topics=10,
        max_num_topics=50,
        step=5,
        iterations=10,
        beta_loss='frobenius',
        algorithm='variational',
        random_state=None,
        verbose=True,
    ):
        cophenetic_correlation = self.topic_model.brunet_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            iterations=iterations,
            beta_loss=beta_loss,
            algorithm=algorithm,
            random_state=random_state,
            verbose=verbose,
        )
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        plt.clf()
        plt.plot(num_topics_infer, cophenetic_correlation, 'o-')
        plt.xticks(num_topics_infer)
        plt.title('Brunet et al. metric (higher is better)')
        plt.xlabel('Number of Topics')
        plt.ylabel('Cophenetic correlation coefficient')
        # find and annotate the maximum point on the plot
        ymax = max(cophenetic_correlation)
        xpos = cophenetic_correlation.index(ymax)
        best_k = num_topics_infer[xpos]
        plt.annotate(f'k={best_k}', xy=(best_k, ymax), xytext=(best_k, ymax), textcoords='offset points', fontsize=16)
        file_path_fig = self.output_dir / 'brunet.png'
        file_path_data = self.output_dir / 'brunet.tsv'
        plt.savefig(file_path_fig)
        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=cophenetic_correlation, step=step, metric_type='brunet')

    def plot_coherence_w2v_metric(
        self,
        min_num_topics=10,
        step=5,
        max_num_topics=50,
        top_n_words=10,
        w2v_size=None,
        w2v_min_count=None,
        # w2v_max_vocab_size=None,
        w2v_max_final_vocab=None,
        w2v_sg=None,
        w2v_workers=None,
        beta_loss='frobenius',
        algorithm='variational',
        random_state=None,
        verbose=True,
    ):
        coherence = self.topic_model.coherence_w2v_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            top_n_words=top_n_words,
            w2v_size=w2v_size,
            w2v_min_count=w2v_min_count,
            # w2v_max_vocab_size=w2v_max_vocab_size,
            w2v_max_final_vocab=w2v_max_final_vocab,
            w2v_sg=w2v_sg,
            w2v_workers=w2v_workers,
            beta_loss=beta_loss,
            algorithm=algorithm,
            random_state=random_state,
            verbose=verbose,
        )

        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        plt.clf()
        # create the line plot
        plt.plot(num_topics_infer, coherence, 'o-')
        plt.xticks(num_topics_infer)
        plt.title('Coherence-Word2Vec metric (higher is better)')
        plt.xlabel('Number of Topics')
        plt.ylabel('Mean Coherence')
        # find and annotate the maximum point on the plot
        ymax = max(coherence)
        xpos = coherence.index(ymax)
        best_k = num_topics_infer[xpos]
        plt.annotate(f'k={best_k}', xy=(best_k, ymax), xytext=(best_k, ymax), textcoords='offset points', fontsize=16)
        file_path_fig = self.output_dir / 'coherence_w2v.png'
        file_path_data = self.output_dir / 'coherence_w2v.tsv'
        plt.savefig(file_path_fig)

        save_topic_number_metrics_data(
            file_path_data,
            range_=(min_num_topics, max_num_topics),
            data=coherence, step=step, metric_type='coherence_w2v')

    def plot_perplexity_metric(
        self,
        min_num_topics=10,
        max_num_topics=20,
        step=5,
        train_size=0.7,
        algorithm='variational',
        random_state=None,
        verbose=True,
    ):
        train_perplexities, test_perplexities = self.topic_model.perplexity_metric(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            step=step,
            train_size=train_size,
            algorithm=algorithm,
            random_state=random_state,
            verbose=verbose,
        )
        num_topics_infer = range(min_num_topics, max_num_topics + 1, step)
        if (len(train_perplexities) > 0) and (len(test_perplexities) > 0):
            plt.clf()
            plt.plot(num_topics_infer, train_perplexities, 'o-', label='Train')
            plt.plot(num_topics_infer, test_perplexities, 'o-', label='Test')
            plt.xticks(num_topics_infer)
            plt.title('Perplexity metric (lower is better)')
            plt.xlabel('Number of Topics')
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

    def plot_docs_over_time(
        self,
        freq: str = '1YS',
        count=True,
        by_affil: bool = False,
        ma_window: int = None,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot count of documents per frequency window, optionally by affiliation
        """

        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
        else:
            groupby = [pd.Grouper(freq=freq)]

        result_count = self.topic_model.corpus.data_frame.reset_index().set_index(
            self.topic_model.corpus._date_col).groupby(
            by=groupby).size()
        if by_affil:
            result_count = result_count.unstack().fillna(0)

        if not count:
            total_count = self.topic_model.corpus.data_frame.reset_index().set_index(
                self.topic_model.corpus._date_col).groupby(
                by=[pd.Grouper(freq=freq)]).size()

            result_count = result_count.div(total_count, axis=0)

        if ma_window:
            result_count = result_count.rolling(window=ma_window, min_periods=1, center=True).mean()

        fig, ax = plt.subplots(figsize=figsize)
        result_count.plot(ax=ax, kind='line')

        if count:
            title_str = 'Document Counts'
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
            title_str = 'Percent of Documents'

        if by_affil:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            title_str += ' Per Affiliation'
        title_str += ' Per Year'

        ax.set_title(title_str)
        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            if count:
                plot_string = 'doc_count'
            else:
                plot_string = 'doc_percent'

            if by_affil:
                affil_string = 'affil'
            else:
                affil_string = 'overall'

            if ma_window:
                ma_string = f'_{ma_window}_MA'
            else:
                ma_string = ''

            filename_out = f'{plot_string}_{affil_string}{ma_string}.{figformat}'

            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, ax, filename_out

    def plot_docs_above_thresh(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        thresh: float = 0.5,
        kind: str = 'count',
        n_words: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot the number of documents associated with each topic, above some threshold
        kind = 'count' or 'percent'
        """

        fig, ax = plt.subplots(figsize=figsize)

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        result = np.array(
            (self.topic_model.topic_distribution_for_document(normalized=normalized) >= thresh).sum(axis=0)
        )[0]

        if kind == 'count':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
            ax.set_ylabel('Count of Documents')
        elif kind == 'percent':
            result = result / self.topic_model.corpus.size
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            ax.set_ylabel('Percent of Documents')

        result = result[[tc in topic_cols for tc in topic_cols_all]]
        sns.barplot(x=topic_cols, y=result, ax=ax)
        # result = pd.DataFrame(data=result, columns=topic_cols_all)[topic_cols]
        # sns.barplot(ax=ax, data=result)

        title_str = f'Documents above {thresh} topic loading'
        if normalized:
            title_str = f'{title_str} ({norm_string})'
        title_str = f'{title_str}; {self.topic_model.corpus.size:,} total docs'

        ax.set_title(title_str)
        fig.autofmt_xdate()

        if savefig:
            plot_string = 'hist_above_thresh'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            filename_out = f'{plot_string}_{topics_string}{norm_string}_{kind}.{figformat}'
            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, ax, filename_out

    def plot_heatmap(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        mask_thresh: float = None,
        cmap=None,
        vmax: float = None,
        vmin: float = None,
        fmt: str = '.2f',
        annot_fontsize: int = 13,
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot a heatmap of topic-topic Pearson correlation coefficient values
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        corr = pd.DataFrame(
            data=np.corrcoef(self.topic_model.topic_distribution_for_document(normalized=normalized).T),
            columns=topic_cols_all,
            index=topic_cols_all,
        )
        corr = corr.loc[topic_cols, topic_cols]

        if mask_thresh is None:
            mask_thresh = 0
        if figsize is None:
            figsize = (max(25, min(len(topic_cols) // 1.1, 25)), max(15, min(len(topic_cols) // 1.2, 15)))
        if cmap is None:
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if vmax is None:
            vmax = corr.max().max()
        # vmax=0.25
        # vmin=-vmax
        if vmin is None:
            vmin = corr.min().min()

        fig, ax = plt.subplots(figsize=figsize)

        ax = sns.heatmap(corr, ax=ax, center=0, annot=True, fmt=fmt, annot_kws={'fontsize': annot_fontsize},
                         vmin=vmin, vmax=vmax,
                         mask=((corr > -mask_thresh) & (corr < mask_thresh)),
                         cmap=cmap,
                         cbar_kws={'label': 'Pearson Correlation Coefficient'},
                         # square=True,
                         )

        ax.hlines(range(1, corr.shape[0]), *ax.get_xlim(), lw=0.5)
        ax.vlines(range(1, corr.shape[1]), *ax.get_ylim(), lw=0.5)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

        if savefig:
            plot_string = 'topic-topic_corr'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            filename_out = f'{plot_string}_{topics_string}{norm_string}.{figformat}'
            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, ax, filename_out

    def plot_clustermap(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        mask_thresh: float = None,
        cmap=None,
        vmax: float = None,
        vmin: float = None,
        fmt: str = '.2f',
        annot_fontsize: int = 13,
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
        metric: str = None,
        method: str = None,
    ):
        """Plot a hierarchical clustermap of topic-topic Pearson correlation coefficient values
        (computed with np.corrcoef). Plot is made with Seaborn's clustermap.
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        corr = pd.DataFrame(
            data=np.corrcoef(self.topic_model.topic_distribution_for_document(normalized=normalized).T),
            columns=topic_cols_all,
            index=topic_cols_all,
        )
        corr = corr.loc[topic_cols, topic_cols]

        if mask_thresh is None:
            mask_thresh = 0
        if figsize is None:
            figsize = (max(25, min(len(topic_cols) // 1.1, 25)), max(15, min(len(topic_cols) // 1.2, 15)))
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
            center=0, annot=True, fmt=fmt, annot_kws={'fontsize': annot_fontsize},
            metric=metric,
            method=method,
            vmin=vmin, vmax=vmax,
            mask=((corr > -mask_thresh) & (corr < mask_thresh)),
            cmap=cmap,
            figsize=figsize,
            cbar_kws={'label': '\n'.join('Pearson Correlation Coefficient'.split())},
        )

        g.ax_heatmap.hlines(range(1, corr.shape[0]), *g.ax_heatmap.get_xlim(), lw=0.5)
        g.ax_heatmap.vlines(range(1, corr.shape[1]), *g.ax_heatmap.get_ylim(), lw=0.5)

        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=30, ha='right', fontsize=18)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=18)

        if savefig:
            plot_string = 'topic-topic_corr_grouped'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            filename_out = f'{plot_string}_{topics_string}{norm_string}'
            filename_out_img = f'{filename_out}.{figformat}'
            filename_out_data = f'{filename_out}.csv'
            # save image to disk
            g.savefig(self.output_dir / filename_out_img, dpi=dpi, transparent=False, bbox_inches='tight')
            # save values to csv
            corr.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind].to_csv(self.output_dir / filename_out_data)
            plt.close('all')
        else:
            filename_out_img = None
            plt.show()

        return g, filename_out_img

    def plot_topic_loading_hist(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        bins=None,
        ncols: int = None,
        n_words: int = 10,
        nchar_title: int = None,
        figsize_scale: int = None,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot histogram of document loading distributions per topic
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if ncols is None:
            ncols = 5
        if ncols > len(topic_cols):
            ncols = len(topic_cols)

        nrows = int(np.ceil(len(topic_cols) / ncols))

        if figsize_scale is None:
            figsize_scale = 3

        if figsize is None:
            figsize = (ncols * figsize_scale, nrows * figsize_scale)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            norm_string = 'normalized'
            if bins is None:
                bins = np.arange(0, 1.05, 0.05)
        else:
            norm_string = ''
            if bins is None:
                bins = 10

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]

        for topic_col, ax in zip(topic_cols, axes.ravel()):
            _df[topic_col].plot(ax=ax, kind='hist', bins=bins)
            title = split_string_nchar(topic_col, nchar=nchar_title)
            ax.set_title(title)
            xlabel = 'Topic Loading'
            if normalized:
                ax.set_xlabel(f'{xlabel}\n({norm_string})')
                ax.set_xlim((0, 1))
            else:
                ax.set_xlabel(xlabel)
            # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
            # ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()]);

        # show xticklabels on all axes
        for topic_col, ax in zip(topic_cols, axes.ravel()):
            plt.setp(ax.get_xticklabels(), visible=True)

        # removed unused axes
        for i in range(len(topic_cols), nrows * ncols):
            axes.ravel()[i].axis('off')

        fig.tight_layout()

        if savefig:
            plot_string = 'topic_loading_hist'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            else:
                norm_string = ''

            filename_out = f'{plot_string}_{topics_string}{norm_string}.{figformat}'

            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, axes, filename_out

    def plot_topic_loading_boxplot(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        ylim: Tuple[float, float] = None,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Marginal distributions of topic loadings

        Plot Boxplot of document loading distributions per topic
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        fig, ax = plt.subplots(figsize=figsize)

        if normalized:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            norm_string = 'normalized'
            ax.set_ylabel(f'Topic Loading ({norm_string})')
        else:
            norm_string = ''
            ax.set_ylabel('Topic Loading (absolute)')

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]
        ax = sns.boxplot(ax=ax, data=_df)

        ax.set_title('Topic Loading Distribution (boxplot)')

        if ylim:
            ax.set_ylim(ylim)

        fig.autofmt_xdate()

        if savefig:
            plot_string = 'topic_loading_boxplot'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            else:
                norm_string = ''

            filename_out = f'{plot_string}_{topics_string}{norm_string}.{figformat}'

            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, ax, filename_out

    def plot_topic_loading_barplot(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        ylim: Tuple[float, float] = None,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Marginal distributions of topic loadings

        Plot Barplot of document loading distributions per topic
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        fig, ax = plt.subplots(figsize=figsize)

        if normalized:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            norm_string = 'normalized'
            ax.set_ylabel(f'Average Topic Loading ({norm_string})')
        else:
            norm_string = ''
            ax.set_ylabel('Average Topic Loading (absolute)')

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]
        ax = sns.barplot(ax=ax, data=_df, estimator=np.mean)

        ax.set_title('Topic Loading Distribution (barplot; 95% CI of the mean)')

        if ylim:
            ax.set_ylim(ylim)

        fig.autofmt_xdate()

        if savefig:
            plot_string = 'topic_loading_barplot'
            topics_string = f'{len(topic_cols)}_topics'
            if normalized:
                norm_string = f'_{norm_string}'
            else:
                norm_string = ''

            filename_out = f'{plot_string}_{topics_string}{norm_string}.{figformat}'

            # save image to disk
            fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')
            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, ax, filename_out

    def plot_one_topic_over_time_count(
        self,
        topic_col: str,
        normalized: bool = True,
        thresh: float = 0.1,
        freq: str = '1YS',
        n_words: int = 10,
    ):

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))

        idx = topic_cols_all.index(topic_col)

        addtl_cols = [self.topic_model.corpus._date_col]

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized)[:, idx],
            columns=[topic_col],
        )
        _df = pd.merge(_df, self.topic_model.corpus.data_frame[addtl_cols], left_index=True, right_index=True)
        _df = _df.reset_index().set_index(self.topic_model.corpus._date_col)

        result = _df[_df[topic_col] >= thresh].groupby(
            pd.Grouper(freq=freq))[topic_col].size()

        if result.empty:
            print(f"No documents >= {thresh}")
            fig = None
            ax = None
        else:
            fig, ax = plt.subplots()
            result.plot(ax=ax, kind='line', marker='o')

            ax.set_title(topic_col)

            ylabel = f"# of year's documents >= {thresh}"
            if normalized:
                # ax.set_ylim((-0.05, 1.05))
                ylabel = f"{ylabel}\n({norm_string})"

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Publication year")
            plt.show()

        return fig, ax

    def plot_one_topic_over_time_percent(
        self,
        topic_col: str,
        normalized: bool = True,
        thresh: float = 0.1,
        freq: str = '1YS',
        n_words: int = 10,
    ):

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))

        idx = topic_cols_all.index(topic_col)

        addtl_cols = [self.topic_model.corpus._date_col]

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized)[:, idx],
            columns=[topic_col],
        )
        _df = pd.merge(_df, self.topic_model.corpus.data_frame[addtl_cols], left_index=True, right_index=True)
        _df = _df.reset_index().set_index(self.topic_model.corpus._date_col)

        result_total = _df.groupby(pd.Grouper(freq=freq))[topic_col].size()
        result_thresh = _df[_df[topic_col] >= thresh].groupby(
            pd.Grouper(freq=freq))[topic_col].size()
        result = result_thresh / result_total

        if result.empty:
            print(f"No documents >= {thresh}")
            fig = None
            ax = None
        else:
            fig, ax = plt.subplots()
            result.plot(ax=ax, kind='line', marker='o')

            ax.set_title(topic_col)

            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

            ylabel = f"% of year's documents >= {thresh}"
            if normalized:
                ylabel = f"{ylabel}\n({norm_string})"

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Publication year")
            plt.show()

        return fig, ax

    def plot_topic_over_time_count(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        thresh: float = 0.1,
        freq: str = '1YS',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window: int = None,
        by_affil: bool = False,
        figsize_scale: int = None,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot count of documents >= a given threshold per frequency window
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]

        if ncols is None:
            ncols = 5
        if ncols > len(topic_cols):
            ncols = len(topic_cols)

        nrows = int(np.ceil(len(topic_cols) / ncols))

        if figsize_scale is None:
            figsize_scale = 3

        if figsize is None:
            figsize = (ncols * figsize_scale, nrows * figsize_scale)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]
        _df = pd.merge(_df, self.topic_model.corpus.data_frame[addtl_cols], left_index=True, right_index=True)
        _df = _df.reset_index().set_index(self.topic_model.corpus._date_col)

        # so all have the same axes
        idx = _df.groupby(
            by=[pd.Grouper(freq=freq),
                self.topic_model.corpus._affiliation_col,
                ])[topic_cols].size().unstack().index

        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
        else:
            groupby = [pd.Grouper(freq=freq)]

        for topic_col, ax in zip(topic_cols, axes.ravel()):
            result_thresh = _df[_df[topic_col] >= thresh].groupby(
                by=groupby)[topic_col].size()
            result = pd.DataFrame(index=idx)
            if by_affil:
                result = result.merge(result_thresh.unstack(), how='outer',
                                      left_index=True, right_index=True).fillna(0)
            else:
                result = result.merge(result_thresh, how='outer',
                                      left_index=True, right_index=True).fillna(0)
            if ma_window:
                result = result.rolling(window=ma_window, min_periods=1, center=True).mean()
            result.plot(ax=ax, kind='line', marker='', legend=None)

            title = split_string_nchar(topic_col, nchar=nchar_title)
            ax.set_title(title)
            ylabel = f"# of year's documents >= {thresh}"
            if normalized:
                ylabel = f"{ylabel}\n({norm_string})"
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Publication year")

        # show xticklabels on all axes
        for topic_col, ax in zip(topic_cols, axes.ravel()):
            plt.setp(ax.get_xticklabels(), visible=True)

        # removed unused axes
        for i in range(len(topic_cols), nrows * ncols):
            axes.ravel()[i].axis('off')

        # for placing the affiliation legend
        if by_affil:
            handles, labels = ax.get_legend_handles_labels()
            bbox_y = 1.0 + ((1.3**(-nrows)) * 0.25)
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, bbox_y), loc='upper center')

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            plot_string = 'topic_count'
            if by_affil:
                affil_string = 'affil'
            else:
                affil_string = 'overall'
            topics_string = f'{len(topic_cols)}_topics'
            thresh_string = f'{int(thresh * 100)}_topicthresh'
            if normalized:
                norm_string = f'_{norm_string}'
            else:
                norm_string = ''
            if ma_window:
                ma_string = f'_{ma_window}_MA'
            else:
                ma_string = ''

            filename_out = f'{plot_string}_{affil_string}_{topics_string}_{thresh_string}{norm_string}{ma_string}.{figformat}'

            # save image to disk
            if by_affil:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight', bbox_extra_artists=(lgd,))
            else:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')

            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, axes, filename_out

    def plot_topic_over_time_percent(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        thresh: float = 0.1,
        freq: str = '1YS',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window: int = None,
        by_affil: bool = False,
        figsize_scale: int = None,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """Plot the percent of documents above the threshold that are above the threshold for each topic, per year.
        Each year across topics adds up to 100%.
        One document can contribute to multiple topics.
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]

        if ncols is None:
            ncols = 5
        if ncols > len(topic_cols):
            ncols = len(topic_cols)

        nrows = int(np.ceil(len(topic_cols) / ncols))

        if figsize_scale is None:
            figsize_scale = 3

        if figsize is None:
            figsize = (ncols * figsize_scale, nrows * figsize_scale)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]
        _df = pd.merge(_df, self.topic_model.corpus.data_frame[addtl_cols], left_index=True, right_index=True)

        # join the date with boolean >= thresh
        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
            result_thresh = _df[[self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]].join(
                _df[topic_cols] >= thresh).reset_index().set_index(
                    self.topic_model.corpus._date_col).groupby(
                        by=groupby)[topic_cols].sum()
        else:
            groupby = [pd.Grouper(freq=freq)]
            result_thresh = _df[[self.topic_model.corpus._date_col]].join(
                _df[topic_cols] >= thresh).reset_index().set_index(
                    self.topic_model.corpus._date_col).groupby(
                        by=groupby)[topic_cols].sum()

        result = result_thresh.div(result_thresh.sum(axis=1), axis='index')

        if ma_window:
            result = result.rolling(window=ma_window, min_periods=1, center=True).mean()

        for topic_col, ax in zip(topic_cols, axes.ravel()):
            if by_affil:
                result[topic_col].unstack().plot(ax=ax, kind='line', marker='', legend=None)
            else:
                result[topic_col].plot(ax=ax, kind='line', marker='', legend=None)

            title = split_string_nchar(topic_col, nchar=nchar_title)
            ax.set_title(title)
            ylabel = f"% of year's documents >= {thresh}"
            if normalized:
                ylabel = f"{ylabel}\n({norm_string})"
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Publication year")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

        # show xticklabels on all axes
        for topic_col, ax in zip(topic_cols, axes.ravel()):
            plt.setp(ax.get_xticklabels(), visible=True)

        # removed unused axes
        for i in range(len(topic_cols), nrows * ncols):
            axes.ravel()[i].axis('off')

        # for placing the affiliation legend
        if by_affil:
            handles, labels = ax.get_legend_handles_labels()
            bbox_y = 1.0 + ((1.3**(-nrows)) * 0.25)
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, bbox_y), loc='upper center')

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            plot_string = 'topic_percent'
            if by_affil:
                affil_string = 'affil'
            else:
                affil_string = 'overall'
            topics_string = f'{len(topic_cols)}_topics'
            thresh_string = f'{int(thresh * 100)}_topicthresh'
            if normalized:
                norm_string = f'_{norm_string}'
            if ma_window:
                ma_string = f'_{ma_window}_MA'
            else:
                ma_string = ''

            filename_out = f'{plot_string}_{affil_string}_{topics_string}_{thresh_string}{norm_string}{ma_string}.{figformat}'

            # save image to disk
            if by_affil:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight', bbox_extra_artists=(lgd,))
            else:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')

            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, axes, filename_out

    def plot_topic_over_time_loading(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        thresh: float = 0.1,
        freq: str = '1YS',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window: int = None,
        by_affil: bool = False,
        figsize_scale: int = None,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        """For each topic (separately), of the documents above the threshold,
        plot the average topic loading for that year.
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]

        if ncols is None:
            ncols = 5
        if ncols > len(topic_cols):
            ncols = len(topic_cols)

        nrows = int(np.ceil(len(topic_cols) / ncols))

        if figsize_scale is None:
            figsize_scale = 3

        if figsize is None:
            figsize = (ncols * figsize_scale, nrows * figsize_scale)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            norm_string = 'normalized'
        else:
            norm_string = ''

        _df = pd.DataFrame(
            data=self.topic_model.topic_distribution_for_document(normalized=normalized),
            columns=topic_cols_all,
        )
        _df = _df[topic_cols]
        _df = pd.merge(_df, self.topic_model.corpus.data_frame[addtl_cols], left_index=True, right_index=True)
        _df = _df.reset_index().set_index(self.topic_model.corpus._date_col)

        # so all have the same axes
        idx = _df.groupby(
            by=[pd.Grouper(freq=freq),
                self.topic_model.corpus._affiliation_col,
                ])[topic_cols].size().unstack().index

        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
        else:
            groupby = [pd.Grouper(freq=freq)]

        for topic_col, ax in zip(topic_cols, axes.ravel()):
            result_thresh = _df[_df[topic_col] >= thresh].groupby(
                by=groupby)[topic_col].mean()
            result = pd.DataFrame(index=idx)
            if by_affil:
                result = result.merge(result_thresh.unstack(), how='outer',
                                      left_index=True, right_index=True).fillna(0)
            else:
                result = result.merge(result_thresh, how='outer',
                                      left_index=True, right_index=True).fillna(0)
            if ma_window:
                result = result.rolling(window=ma_window, min_periods=1, center=True).mean()
            result.plot(ax=ax, kind='line', marker='', legend=None)

            title = split_string_nchar(topic_col, nchar=nchar_title)
            ax.set_title(title)
            ylabel = "Avg. Topic Load per Year"
            if normalized:
                ylabel = f"{ylabel}\n({norm_string})"
                ax.set_ylim((-0.05, 1.05))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Publication year")

        # show xticklabels on all axes
        for topic_col, ax in zip(topic_cols, axes.ravel()):
            plt.setp(ax.get_xticklabels(), visible=True)

        # removed unused axes
        for i in range(len(topic_cols), nrows * ncols):
            axes.ravel()[i].axis('off')

        # for placing the affiliation legend
        if by_affil:
            handles, labels = ax.get_legend_handles_labels()
            bbox_y = 1.0 + ((1.3**(-nrows)) * 0.25)
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, bbox_y), loc='upper center')

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            plot_string = 'topic_loading'
            if by_affil:
                affil_string = 'affil'
            else:
                affil_string = 'overall'
            topics_string = f'{len(topic_cols)}_topics'
            thresh_string = f'{int(thresh * 100)}_topicthresh'
            if normalized:
                norm_string = '_normalized'
            else:
                norm_string = ''
            if ma_window:
                ma_string = f'_{ma_window}_MA'
            else:
                ma_string = ''

            filename_out = f'{plot_string}_{affil_string}_{topics_string}_{thresh_string}{norm_string}{ma_string}.{figformat}'

            # save image to disk
            if by_affil:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight', bbox_extra_artists=(lgd,))
            else:
                fig.savefig(self.output_dir / filename_out, dpi=dpi, transparent=False, bbox_inches='tight')

            plt.close('all')
        else:
            filename_out = None
            plt.show()

        return fig, axes, filename_out

    def plotly_docs_over_time(
        self,
        freq: str = '1YS',
        count=True,
        by_affil: bool = False,
        ma_window: int = None,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Line plot of the count of documents per frequency window.
        Optionally by affiliation.
        Optionally as a percent (adds to 100% for a given frequency window).
        """
        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
        else:
            groupby = [pd.Grouper(freq=freq)]

        result_count = self.topic_model.corpus.data_frame.reset_index().set_index(
            self.topic_model.corpus._date_col).groupby(
            by=groupby).size()
        if by_affil:
            result_count = result_count.unstack().fillna(0)

        if not count:
            total_count = self.topic_model.corpus.data_frame.reset_index().set_index(
                self.topic_model.corpus._date_col).groupby(
                by=[pd.Grouper(freq=freq)]).size()

            result_count = result_count.div(total_count, axis=0)

        if ma_window:
            result_count = result_count.rolling(window=ma_window, min_periods=1, center=True).mean()

        xlabel = 'Date'
        if count:
            title_str = 'Document Counts'
            ylabel = 'Count'
            autorange = True
            tickformat = None
            yrange = None
        else:
            title_str = 'Percent of Documents'
            ylabel = 'Percent'
            autorange = False
            tickformat = '.1%'
            yrange = [0, 1]

        if by_affil:
            title_str += ' per Affiliation'
        title_str += ' per Year'

        if savedata:
            save_data_string = title_str.lower().replace(' ', '_')
            filename_out = f'{save_data_string}.csv'
            # save data to disk
            result_count.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        if by_affil:
            affils = self.topic_model.corpus.data_frame[self.topic_model.corpus._affiliation_col].unique()
            data = []
            for affil in affils:
                data.append(
                    go.Scatter(
                        x=result_count.index,
                        y=result_count[affil],
                        mode='lines+markers',
                        name=affil,
                    ),
                )
        else:
            data = [
                go.Scatter(
                    x=result_count.index,
                    y=result_count.values,
                    mode='lines+markers',
                ),
            ]

        layout = go.Layout(
            title=title_str,
            xaxis=go.layout.XAxis(
                title=xlabel,
                tickangle=-30,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
                showticklabels=True,
                ticks='outside',
                nticks=len(result_count.index),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
                autorange=autorange,
                automargin=True,
                tickformat=tickformat,
                range=yrange,
            ),
            margin=go.layout.Margin(
                t=30,
                b=0,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=True,
            width=800,
            height=300,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out

    def plotly_topic_over_time(
        self,
        topic_id: int,
        # freq: str = '1YS',
        count=True,
        # by_affil: bool = False,
        # ma_window: int = None,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Line plot of the count of documents per year.
        Optionally as a percent (adds to 100% across all topics for a given year).
        """

        # doc_ids = self.topic_model.documents_for_topic(topic_id)
        # new_col = 'topic_doc'
        # _df = self.topic_model.corpus.data_frame[['date']]
        # _df = _df.merge(
        #     pd.DataFrame(index=self.topic_model.corpus.data_frame.index, columns=[new_col], data=0),
        #     left_index=True, right_index=True,
        # )
        # _df.loc[doc_ids, new_col] = 1
        # _df = _df.set_index('date').groupby(by=[pd.Grouper(freq=freq)]).mean()
        # years = _df.index
        # frequency = _df[new_col]

        min_year = self.topic_model.corpus.data_frame['date'].dt.year.min()
        max_year = self.topic_model.corpus.data_frame['date'].dt.year.max()
        years = list(range(min_year, max_year + 1))
        frequency = [self.topic_model.topic_frequency(topic_id, year=year, count=count) for year in years]

        xlabel = 'Year'
        if count:
            title_str = 'Document Counts'
            ylabel = 'Count'
            autorange = True
            tickformat = None
            yrange = None
        else:
            title_str = 'Percent of Documents'
            ylabel = 'Percent'
            autorange = False
            tickformat = '.1%'
            yrange = [0, 1]

        # if by_affil:
        #     title_str += ' per Affiliation'

        title_str += ' per Year'

        if savedata:
            _df = pd.DataFrame(data=frequency, columns=[ylabel.lower()], index=years)
            save_data_string = title_str.lower().replace(' ', '_')
            filename_out = f'{save_data_string}_t{topic_id}.csv'
            # save data to disk
            _df.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        data = [
            go.Scatter(
                x=years,
                y=frequency,
                connectgaps=True,
                mode='lines+markers+text',
                fillcolor='#ff7f0e',
            )
        ]

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                title=xlabel,
                tickangle=-30,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
                showticklabels=True,
                ticks='outside',
                nticks=len(years),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                autorange=autorange,
                automargin=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
                tickformat=tickformat,
                range=yrange,
            ),
            margin=go.layout.Margin(
                t=30,
                b=0,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=False,
            width=800,
            height=300,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out

    def plotly_topic_word_weight(
        self,
        topic_id: int,
        normalized: bool = True,
        n_words: int = 20,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Bar plot of the top word weights for a given topic.
        """
        weighted_words = self.topic_model.top_words(topic_id=topic_id, normalized=normalized, num_words=n_words)
        top_words = [w[0] for w in weighted_words]
        word_weights = [w[1] for w in weighted_words]

        ylabel = 'Word Weight'
        save_data_string = ylabel.lower().replace(' ', '_')
        save_data_string = f'{save_data_string}_t{topic_id}'

        if normalized:
            ylabel = f"{ylabel} (normalized)"
            save_data_string = f'{save_data_string}_norm'

        if savedata:
            _df = pd.DataFrame(data=[word_weights], columns=top_words)
            filename_out = f'{save_data_string}.csv'
            # save data to disk
            _df.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        data = [
            go.Bar(
                x=top_words,
                y=word_weights,
                text=top_words,
                textposition='auto',
                marker=dict(
                    color='rgb(49, 130, 189)',
                    # color='rgb(55, 83, 109)',
                ),
                # hoverlabel=dict(
                #     bgcolor='black',
                #     font=dict(
                #         color='white',
                #     )
                # ),
            )
        ]

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                automargin=True,
                tickangle=-30,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                automargin=True,
                autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            margin=go.layout.Margin(
                t=30,
                b=0,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=False,
            width=800,
            height=300,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out

    def plotly_doc_topic_loading(
        self,
        doc_id: int = None,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Bar plot of the topic loadings for a given document.
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if doc_id is None:
            data = [self.topic_model.topic_distribution_for_document(doc_id=doc_id, normalized=normalized).mean(axis=0)]
            ylabel = 'Average Topic Loading'
        else:
            data = [self.topic_model.topic_distribution_for_document(doc_id=doc_id, normalized=normalized)]
            ylabel = 'Document Topic Loading'

        save_data_string = ylabel.lower().replace(' ', '_')
        if doc_id is not None:
            save_data_string = f'{save_data_string}_d{doc_id}'
        if normalized:
            ylabel = f'{ylabel} (normalized)'
            save_data_string = f'{save_data_string}_norm'

        _df = pd.DataFrame(
            data=data,
            columns=topic_cols_all,
        )[topic_cols]

        if savedata:
            filename_out = f'{save_data_string}.csv'
            # save data to disk
            _df.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        x = [f'Topic {i}: {words}' for i, words in enumerate(topic_cols)]

        y = [np.round(v, decimals=5) for v in _df[topic_cols].values.tolist()[0]]
        y_text = [np.round(v, decimals=3) for v in y]
        data = [
            go.Bar(
                x=x,
                y=y,
                text=y_text,
                textposition='auto',
                marker=dict(
                    color='rgb(49, 130, 189)',
                ),
                # opacity=0.85,
            )
        ]

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                tickangle=-30,
                automargin=True,
                # autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                automargin=True,
                autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            margin=go.layout.Margin(
                t=30,
                b=n_words * 32,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=True,
            width=1200,
            height=600,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            # return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

            return plotly.offline.plot(
                figure,
                # config={"displayModeBar": False},
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out
        # else:
        #     from plotly.offline import init_notebook_mode
        #     init_notebook_mode(connected=True)
        #     return plotly.offline.plot(
        #         figure,
        #         # config={"displayModeBar": False},
        #         show_link=False,
        #         include_plotlyjs=False,
        #         # output_type='div',
        #     )

    def plotly_word_topic_loading(
        self,
        word_id: int,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Bar plot of the topic loadings for a given word.
        """
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        _df = pd.DataFrame(
            data=[self.topic_model.topic_distribution_for_word(word_id=word_id, normalized=normalized)],
            columns=topic_cols_all,
        )[topic_cols]

        x = [f'Topic {i}: {words}' for i, words in enumerate(topic_cols)]

        y = [np.round(v, decimals=5) for v in _df[topic_cols].values.tolist()[0]]
        y_text = [np.round(v, decimals=3) for v in y]
        ylabel = 'Word Topic Loading'

        save_data_string = ylabel.lower().replace(' ', '_')
        save_data_string = f'{save_data_string}_w{word_id}'
        if normalized:
            ylabel = f'{ylabel} (normalized)'
            save_data_string = f'{save_data_string}_norm'

        if savedata:
            filename_out = f'{save_data_string}.csv'
            # save data to disk
            _df.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        data = [
            go.Bar(
                x=x,
                y=y,
                text=y_text,
                textposition='auto',
                marker=dict(
                    color='rgb(49, 130, 189)',
                ),
                # opacity=0.85,
            )
        ]

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                tickangle=-30,
                automargin=True,
                # autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                automargin=True,
                autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            margin=go.layout.Margin(
                t=30,
                b=n_words * 32,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=True,
            width=1200,
            height=600,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            # return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

            return plotly.offline.plot(
                figure,
                # config={"displayModeBar": False},
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out
        # else:
        #     from plotly.offline import init_notebook_mode
        #     init_notebook_mode(connected=True)
        #     return plotly.offline.plot(
        #         figure,
        #         # config={"displayModeBar": False},
        #         show_link=False,
        #         include_plotlyjs=False,
        #         # output_type='div',
        #     )

    def plotly_topic_affiliation_count(
        self,
        topic_id: int,
        output_type: str = 'div',
        savedata: bool = False,
    ):
        """Bar plot of the counts of affiliations of the documents
        for which a given topic is the most likely topic.
        """
        affiliation_count = self.topic_model.affiliation_count(topic_id)
        affiliations = [x[0] for x in affiliation_count]
        counts = [x[1] for x in affiliation_count]

        ylabel = 'Document Count'

        if savedata:
            _df = pd.DataFrame(data=[counts], columns=affiliations)
            save_data_string = 'affiliation_document_count'
            filename_out = f'{save_data_string}_t{topic_id}.csv'
            # save data to disk
            _df.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        data = [
            go.Bar(
                x=affiliations,
                y=counts,
                text=counts,
                textposition='auto',
                marker=dict(
                    color='rgb(49, 130, 189)',
                ),
            )
        ]

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                tickangle=-30,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                automargin=True,
                autorange=True,
                tickfont=dict(
                    size=10,
                    color='rgb(107, 107, 107)',
                ),
            ),
            margin=go.layout.Margin(
                t=30,
                b=0,
                l=30,
                r=0,
                pad=4,
            ),
            autosize=True,
            width=500,
            height=300,
        )

        figure = go.Figure(data=data, layout=layout)

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out

    def plotly_heatmap(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        output_type: str = 'div',
        colorscale: str = None,
        zmax: float = None,
        zmin: float = None,
        zhoverformat='.3f',
        annotate: bool = True,
        annot_decimals: int = 2,
        annot_fontsize: int = 12,
        annot_fontcolor: str = None,
        savedata: bool = False,
    ):
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if colorscale is None:
            # colorscale = 'RdBu'
            # https://www.plotly.express/plotly_express/colors/index.html
            colorscale = px.colors.diverging.RdBu

        if zmax is None or zmin is None:
            zauto = True
        else:
            zauto = False

        corr = pd.DataFrame(
            data=np.corrcoef(self.topic_model.topic_distribution_for_document(normalized=normalized).T),
            columns=topic_cols_all,
            index=topic_cols_all,
        )
        corr = corr.loc[topic_cols, topic_cols]

        save_data_string = 'topic_topic_corr_heatmap'
        if normalized:
            save_data_string = f'{save_data_string}_norm'

        if savedata:
            filename_out = f'{save_data_string}.csv'
            # save data to disk
            corr.to_csv(self.output_dir / filename_out)
        else:
            filename_out = None

        z = corr.values
        x = corr.columns.tolist()
        y = corr.index.tolist()

        if annotate:
            z_text = np.round(z, decimals=annot_decimals)
        else:
            z_text = None

        figure = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=z_text,
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            zmax=zmax,
            zmin=zmin,
            zmid=0,
            zauto=zauto,
            zhoverformat=zhoverformat,
            colorbar={
                'title': {
                    'text': 'Pearson Correlation Coefficient',
                    'side': 'right',
                },
            },
        )

        for annotation in figure['layout']['annotations']:
            if annot_fontsize:
                annotation['font']['size'] = annot_fontsize
            if annot_fontcolor:
                annotation['font']['color'] = annot_fontcolor

        figure.update_layout(
            {
                # 'title': 'Correlation',
                'xaxis': go.layout.XAxis(
                    tickangle=-30,
                    tickfont=dict(
                        size=10,
                        color='rgb(107, 107, 107)'
                    ),
                ),
                'yaxis': go.layout.YAxis(
                    tickfont=dict(
                        size=10,
                        color='rgb(107, 107, 107)'
                    ),
                    autorange='reversed',
                ),
                # 'margin': go.layout.Margin(
                #     t=30,
                #     b=0,
                #     l=30,
                #     r=0,
                #     pad=4,
                # ),
                # 'autosize': True,
                'width': 1200,
                'height': 1000,
                'showlegend': False,
                'hovermode': 'closest',
            }
        )

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out
        else:
            return figure, filename_out

    def plotly_clustermap(
        self,
        topic_cols: List[str] = None,
        normalized: bool = True,
        n_words: int = 10,
        output_type: str = 'div',
        colorscale: str = None,
        zmax: float = None,
        zmin: float = None,
        zhoverformat='.3f',
        annotate: bool = True,
        annot_decimals: int = 2,
        annot_fontsize: int = 12,
        annot_fontcolor: str = None,
        savedata: bool = False,
    ):
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if colorscale is None:
            # colorscale = 'RdBu'
            # https://www.plotly.express/plotly_express/colors/index.html
            colorscale = px.colors.diverging.RdBu

        if zmax is None or zmin is None:
            zauto = True
        else:
            zauto = False

        corr = pd.DataFrame(
            data=np.corrcoef(self.topic_model.topic_distribution_for_document(normalized=normalized).T),
            columns=topic_cols_all,
            index=topic_cols_all,
        )
        corr = corr.loc[topic_cols, topic_cols]

        z = corr.values
        x = corr.columns
        y = corr.index

        # Initialize figure by creating upper dendrogram
        figure = ff.create_dendrogram(z, orientation='bottom', labels=x)
        for i in range(len(figure['data'])):
            figure['data'][i]['yaxis'] = 'y2'

        # Create Side Dendrogram
        dendro_side = ff.create_dendrogram(z, orientation='right')
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        # Add Side Dendrogram Data to Figure
        for data in dendro_side['data']:
            figure.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))

        # Reorder columns and rows
        x = list(x[dendro_leaves])
        y = list(y[dendro_leaves])

        z = z[dendro_leaves, :]
        z = z[:, dendro_leaves]

        if savedata:
            save_data_string_cm = 'topic_topic_corr_clustermap'
            save_data_string_hm = 'topic_topic_corr_heatmap'
            if normalized:
                save_data_string_cm = f'{save_data_string_cm}_norm'
                save_data_string_hm = f'{save_data_string_hm}_norm'
            filename_out_cm = f'{save_data_string_cm}.csv'
            filename_out_hm = f'{save_data_string_hm}.csv'

            # save data to disk - clustermap
            _df = pd.DataFrame(data=z, columns=x, index=y)
            _df.to_csv(self.output_dir / filename_out_cm)
            # save data to disk - heatmap
            corr.to_csv(self.output_dir / filename_out_hm)
        else:
            filename_out_cm = None
            filename_out_hm = None

        if annotate:
            z_text = np.round(z, decimals=annot_decimals)
        else:
            z_text = None

        hm_fig = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=z_text,
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            zmax=zmax,
            zmin=zmin,
            zmid=0,
            zauto=zauto,
            zhoverformat=zhoverformat,
            colorbar={
                'x': -0.15,
                'title': {
                    'text': 'Pearson Correlation Coefficient',
                    'side': 'right',
                },
            },
        )
        # Get the data from the heatmap
        data = hm_fig['data'][0]

        # Get the annotations from the heatmap
        annotations = hm_fig['layout']['annotations']

        # Replace annotation x- and y-axis string tick values with numerical locations
        x_dict = {topic: x for topic, x in zip(data['x'], figure['layout']['xaxis']['tickvals'])}
        y_dict = {topic: y for topic, y in zip(data['y'], dendro_side['layout']['yaxis']['tickvals'])}
        for annotation in annotations:
            annotation['x'] = x_dict[annotation['x']]
            annotation['y'] = y_dict[annotation['y']]
            if annot_fontsize:
                annotation['font']['size'] = annot_fontsize
            if annot_fontcolor:
                annotation['font']['color'] = annot_fontcolor

        # Replace x- and y-axis string tick values with numerical locations
        data['x'] = figure['layout']['xaxis']['tickvals']
        data['y'] = dendro_side['layout']['yaxis']['tickvals']

        # Add Heatmap Data to Figure
        figure.add_trace(data)

        figure.update_layout(
            {
                # 'title': 'Correlation',
                'width': 1400,
                'height': 1100,
                'showlegend': False,
                'hovermode': 'closest',
                'annotations': annotations,
            },
        )

        figure.update_layout(
            xaxis={
                'domain': [0.15, 1.0],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticks': '',
                'tickangle': 30,
                'tickfont': dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
            })

        figure.update_layout(
            xaxis2={
                'domain': [0, 0.15],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': '',
            })

        figure.update_layout(
            yaxis={
                'domain': [0, 0.85],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticks': '',
                'tickmode': 'array',
                'tickvals': dendro_side['layout']['yaxis']['tickvals'],
                'ticktext': y,
                'tickfont': dict(
                    size=10,
                    color='rgb(107, 107, 107)'
                ),
                'side': 'right',
                'autorange': 'reversed',
            })

        figure.update_layout(
            yaxis2={
                'domain': [0.85, 1.0],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': '',
            })

        if output_type == 'div':
            return plotly.offline.plot(
                figure,
                show_link=False,
                include_plotlyjs=False,
                output_type=output_type,
            ), filename_out_cm, filename_out_hm
        else:
            return figure, filename_out_cm, filename_out_hm
