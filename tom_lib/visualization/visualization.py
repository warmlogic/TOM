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

from tom_lib.utils import save_topic_number_metrics_data

sns.set(rc={"lines.linewidth": 2})
sns.set_style("whitegrid")

mpl.use("Agg")  # To be able to create figures on a headless server (no DISPLAY variable)


def split_string_sep(string: str, sep: str = None):
    '''Split a string on spaces and put together with newlines
    '''
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
    '''Split a string into a given number of chunks based on number of characters
    '''
    if nchar is None:
        nchar = 25
    return '\n'.join([string[(i * nchar):(i + 1) * nchar] for i in range(int(np.ceil(len(string) / nchar)))])


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

    def plot_docs_over_time_count(
        self,
        freq: str = '1Y',
        by_affil=False,
        ma_window=None,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Plot count of documents per frequency window, optionally by affiliation
        '''

        fig, ax = plt.subplots(figsize=figsize)

        if by_affil:
            groupby = [pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]
        else:
            groupby = [pd.Grouper(freq=freq)]

        result_count = self.topic_model.corpus.data_frame.reset_index().set_index(
            self.topic_model.corpus._date_col).groupby(
            by=groupby).size()

        if ma_window:
            result_count = result_count.rolling(window=ma_window, min_periods=1, center=True).mean()

        if by_affil:
            result_count.unstack().plot(ax=ax, kind='line')
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        else:
            result_count.plot(ax=ax, kind='line')

        ax.set_title('Document counts')

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            plot_string = 'doc_count'
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

    def plot_docs_over_time_percent_affil(
        self,
        freq: str = '1Y',
        ma_window=None,
        figsize: Tuple[int, int] = (12, 8),
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Plot percent of documents per affiliation and frequency window
        '''

        total_count = self.topic_model.corpus.data_frame.reset_index().set_index(
            self.topic_model.corpus._date_col).groupby(
            by=[pd.Grouper(freq=freq)]).size()

        result_count = self.topic_model.corpus.data_frame.reset_index().set_index(
            self.topic_model.corpus._date_col).groupby(
            by=[pd.Grouper(freq=freq), self.topic_model.corpus._affiliation_col]).size() / total_count

        if ma_window:
            result_count = result_count.rolling(window=ma_window, min_periods=1, center=True).mean()

        fig, ax = plt.subplots(figsize=figsize)

        result_count.unstack().plot(ax=ax, kind='line')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        ax.set_title('Percent of documents per affiliation per year')

        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
        fig.tight_layout()

        if savefig:
            plot_string = 'doc_percent_affil'
            if ma_window:
                ma_string = f'_{ma_window}_MA'
            else:
                ma_string = ''
            filename_out = f'{plot_string}{ma_string}.{figformat}'

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
        '''Plot the number of documents associated with each topic, above some threshold
        kind = 'count' or 'percent'
        '''

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

        if kind == 'count':
            if normalized:
                result = ((
                    self.topic_model.document_topic_matrix / self.topic_model.document_topic_matrix.sum(axis=1)) >= thresh).sum(axis=0)
            else:
                result = (self.topic_model.document_topic_matrix >= thresh).sum(axis=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
            ax.set_ylabel('Count of documents')
        elif kind == 'percent':
            if normalized:
                result = ((
                    (self.topic_model.document_topic_matrix / self.topic_model.document_topic_matrix.sum(axis=1)) >= thresh).sum(axis=0) / self.topic_model.corpus.size)
            else:
                result = ((
                    self.topic_model.document_topic_matrix >= thresh).sum(axis=0) / self.topic_model.corpus.size)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            ax.set_ylabel('Percent of documents')

        result = np.array(result)[0]
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
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
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
            norm_string = 'normalized'
        else:
            corr = np.corrcoef(self.topic_model.document_topic_matrix.todense().T)
            norm_string = ''

        corr = pd.DataFrame(data=corr, columns=topic_cols_all, index=topic_cols_all)
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

        ax = sns.heatmap(corr, ax=ax, center=0, annot=True, fmt=fmt, annot_kws={"fontsize": 13},
                         vmin=vmin, vmax=vmax,
                         mask=((corr > -mask_thresh) & (corr < mask_thresh)),
                         cmap=cmap,
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
        n_words: int = 10,
        figsize: Tuple[int, int] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
        metric: str = None,
        method: str = None,
    ):
        '''Plot a clustermap of a correlation dataframe (df.corr())
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
            norm_string = 'normalized'
        else:
            corr = np.corrcoef(self.topic_model.document_topic_matrix.todense().T)
            norm_string = ''

        corr = pd.DataFrame(data=corr, columns=topic_cols_all, index=topic_cols_all)
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
            center=0, annot=True, fmt=fmt, annot_kws={"fontsize": 13},
            metric=metric,
            method=method,
            vmin=vmin, vmax=vmax,
            mask=((corr > -mask_thresh) & (corr < mask_thresh)),
            cmap=cmap,
            figsize=figsize,
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
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Plot histogram of document loading distributions per topic
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if ncols is None:
            ncols = 5
        nrows = int(np.ceil(len(topic_cols) / ncols))

        fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all)
            norm_string = 'normalized'
            if bins is None:
                bins = np.arange(0, 1.05, 0.05)
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all)
            norm_string = ''
            if bins is None:
                bins = 10

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
        figsize: Tuple[int, int] = (12, 8),
        n_words: int = 10,
        ylim: Tuple[float, float] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Marginal distributions of topic loadings

        Plot Boxplot of document loading distributions per topic
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        fig, ax = plt.subplots(figsize=figsize)

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            norm_string = 'normalized'
            ax.set_ylabel(f'Topic Loading ({norm_string})')
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all)
            norm_string = ''
            ax.set_ylabel('Topic Loading (absolute)')

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
        figsize: Tuple[int, int] = (12, 8),
        n_words: int = 10,
        ylim: Tuple[float, float] = None,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Marginal distributions of topic loadings

        Plot Barplot of document loading distributions per topic
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        fig, ax = plt.subplots(figsize=figsize)

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            norm_string = 'normalized'
            ax.set_ylabel(f'Average Topic Loading ({norm_string})')
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all)
            norm_string = ''
            ax.set_ylabel('Average Topic Loading (absolute)')

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
        freq: str = '1Y',
        n_words: int = 10,
    ):

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))

        idx = topic_cols_all.index(topic_col)

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1))[:, idx],
                columns=[topic_col],
            )
            norm_string = 'normalized'
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix[:, idx],
                columns=[topic_col],
            )
            norm_string = ''

        addtl_cols = [self.topic_model.corpus._date_col]
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
        freq: str = '1Y',
        n_words: int = 10,
    ):

        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))

        idx = topic_cols_all.index(topic_col)

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1))[:, idx],
                columns=[topic_col],
            )
            norm_string = 'normalized'
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix[:, idx],
                columns=[topic_col],
            )
            norm_string = ''

        addtl_cols = [self.topic_model.corpus._date_col]
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
        freq: str = '1Y',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window=None,
        by_affil=False,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Plot count of documents >= a given threshold per frequency window
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if ncols is None:
            ncols = 5
        nrows = int(np.ceil(len(topic_cols) / ncols))

        fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all,
            )
            norm_string = 'normalized'
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all,
            )
            norm_string = ''

        _df = _df[topic_cols]

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]
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
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='upper center')

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
        freq: str = '1Y',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window=None,
        by_affil=False,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''Plot the percent of documents that are above the threshold for that year.
        Therefore, a given year across topics adds up to 100%.
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if ncols is None:
            ncols = 5
        nrows = int(np.ceil(len(topic_cols) / ncols))

        fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all,
            )
            norm_string = 'normalized'
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all,
            )
            norm_string = ''

        _df = _df[topic_cols]

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]
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
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='upper center')

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
        freq: str = '1Y',
        n_words: int = 10,
        nchar_title: int = None,
        ncols: int = None,
        ma_window=None,
        by_affil=False,
        savefig: bool = False,
        dpi: int = 72,
        figformat: str = 'png',
    ):
        '''For each topic (separately), of the documents above the threshold,
        plot the average topic loading for that year.
        '''
        topic_cols_all = []
        for top_words in self.topic_model.top_words_topics(n_words):
            topic_cols_all.append(' '.join(top_words))
        if not topic_cols:
            topic_cols = topic_cols_all

        if ncols is None:
            ncols = 5
        nrows = int(np.ceil(len(topic_cols) / ncols))

        fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            nrows=nrows, ncols=ncols,
            sharey=True,
            sharex=True,
        )

        if normalized:
            _df = pd.DataFrame(
                data=(self.topic_model.document_topic_matrix /
                      self.topic_model.document_topic_matrix.sum(axis=1)),
                columns=topic_cols_all,
            )
            norm_string = 'normalized'
        else:
            _df = pd.DataFrame(
                data=self.topic_model.document_topic_matrix.todense(),
                columns=topic_cols_all,
            )
            norm_string = ''

        _df = _df[topic_cols]

        addtl_cols = [self.topic_model.corpus._date_col, self.topic_model.corpus._affiliation_col]
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
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='upper center')

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
