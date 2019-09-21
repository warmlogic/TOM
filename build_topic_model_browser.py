# coding: utf-8
import argparse
import configparser
from math import ceil
import os
from pathlib import Path
import tom_lib.utils as ut
from flask import Flask, render_template, request, send_from_directory
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table as dt
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization
import logging
import urllib

logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)


def get_parser():
    """
    Creates a new argument parser
    """
    parser = argparse.ArgumentParser(prog='run_browser')
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


def main(config_browser):
    # Data parameters
    data_dir = config_browser.get('data_dir', '', vars=os.environ)
    data_dir = data_dir or '.'
    data_dir = Path(data_dir)
    docs_filename = config_browser.get('docs_filename', '')
    if not docs_filename:
        raise ValueError(f'docs_filename not specified in {config_filepath}')
    source_filepath = data_dir / docs_filename
    if not source_filepath.exists():
        raise OSError(f'Documents file does not exist: {source_filepath}')
    # Corpus parameters
    corpus_name = config_browser.get('corpus_name', '')
    corpus_name = corpus_name or 'corpus'
    corpus_name = '_'.join(corpus_name.split())  # remove spaces
    language = config_browser.get('language', None)
    assert (isinstance(language, str) and language in ['english']) or (isinstance(language, list)) or (language is None)
    # ignore words which relative frequency is > than max_relative_frequency
    max_relative_frequency = config_browser.getfloat('max_relative_frequency', 0.8)
    # ignore words which absolute frequency is < than min_absolute_frequency
    min_absolute_frequency = config_browser.getint('min_absolute_frequency', 5)
    # 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
    vectorization = config_browser.get('vectorization', 'tfidf')
    n_gram = config_browser.getint('n_gram', 1)
    max_features = config_browser.get('max_features', None)
    if isinstance(max_features, str):
        if max_features.isnumeric():
            max_features = int(max_features)
        elif max_features == 'None':
            max_features = None
    assert isinstance(max_features, int) or (max_features is None)
    sample = config_browser.getfloat('sample', 1.0)
    # General model parameters
    model_type = config_browser.get('model_type', 'NMF')
    num_topics = config_browser.getint('num_topics', 15)
    verbose = config_browser.getint('verbose', 0)
    random_state = config_browser.getint('random_state', None)
    rename_topics = config_browser.get('rename_topics', None)
    rename_topics = rename_topics.split(',') if rename_topics else None
    merge_topics = config_browser.get('merge_topics', None)
    if merge_topics:
        merge_topics = {t.split(':')[0]: t.split(':')[1:][0].split(',') for t in merge_topics.split('.') if t}
    # must define the state if renaming or merging topics
    if rename_topics or merge_topics:
        assert random_state is not None
    load_if_existing_model = config_browser.getboolean('load_if_existing_model', True)
    # NMF parameters
    nmf_init = config_browser.get('nmf_init', None)
    nmf_solver = config_browser.get('nmf_solver', None)
    nmf_beta_loss = config_browser.get('nmf_beta_loss', 'frobenius')
    nmf_max_iter = config_browser.getint('nmf_max_iter', None)
    nmf_alpha = config_browser.getfloat('nmf_alpha', None)
    nmf_l1_ratio = config_browser.getfloat('nmf_l1_ratio', None)
    nmf_shuffle = config_browser.getboolean('nmf_shuffle', None)
    # LDA parameters
    lda_algorithm = config_browser.get('lda_algorithm', 'variational')
    lda_alpha = config_browser.getfloat('lda_alpha', None)
    lda_eta = config_browser.getfloat('lda_eta', None)
    lda_learning_method = config_browser.get('lda_algorithm', 'batch')
    lda_n_jobs = config_browser.getint('lda_n_jobs', -1)
    lda_n_iter = config_browser.getint('lda_n_iter', None)
    # Web app parameters
    top_words_description = config_browser.getint('top_words_description', 10)
    top_words_cloud = config_browser.getint('top_words_cloud', 5)

    if model_type not in ['NMF', 'LDA']:
        raise ValueError(f"model_type must be 'NMF' or 'LDA', got {model_type}")

    if model_type == 'NMF':
        if (nmf_solver == 'mu') and (nmf_beta_loss not in ['frobenius', 'kullback-leibler', 'itakura-saito']):
            raise ValueError(f"For NMF, 'beta_loss' must be 'frobenius', 'kullback-leibler', or 'itakura-saito', got '{nmf_beta_loss}'")
        if vectorization == 'tf':
            raise ValueError(f"for NMF, 'vectorization' should be 'tfidf', got '{vectorization}'")
    elif model_type == 'LDA':
        if lda_algorithm not in ['variational', 'gibbs']:
            raise ValueError(f"For LDA, 'lda_algorithm' must be 'variational' or 'gibbs', got '{lda_algorithm}'")
        if vectorization == 'tfidf':
            raise ValueError(f"for LDA, 'vectorization' should be 'tf', got '{vectorization}'")

    if rename_topics:
        assert len(rename_topics) == num_topics

    # Flask Web server
    static_folder = Path('browser/static')
    template_folder = Path('browser/templates')

    # Set up directories for serving files
    tm_folder = Path('data') / f'{model_type}_{source_filepath.stem}_{num_topics}_topics'
    data_folder = tm_folder / 'data'
    model_folder = tm_folder / 'model'
    topic_model_filepath = model_folder / 'model.pickle'

    # Set up sub-directories for serving files
    topic_cloud_folder = data_folder / 'topic_cloud'
    # # author_network_folder = data_folder / 'author_network'
    figs_folder = data_folder / 'figs'

    full_text_col = 'orig_text'
    id_col = 'access_num'

    # ##################################
    # Load or train model
    # ##################################

    if load_if_existing_model and (static_folder / topic_model_filepath).exists():
        # Load model from disk:
        logger.info(f'Loading topic model: {static_folder / topic_model_filepath}')
        topic_model = ut.load_topic_model(static_folder / topic_model_filepath)

        logger.info(f'Corpus size: {topic_model.corpus.size:,}')
        logger.info(f'Vocabulary size: {topic_model.corpus.vocabulary_size:,}')
    else:
        # Clean the topic model directory
        if (static_folder / tm_folder).exists():
            ut.delete_folder(static_folder / tm_folder)
        (static_folder / tm_folder).mkdir(parents=True, exist_ok=True)

        # Load and prepare a corpus
        logger.info(f'Loading documents: {source_filepath}')
        corpus = Corpus(
            source_filepath=source_filepath,
            name=corpus_name,
            language=language,
            vectorization=vectorization,
            n_gram=n_gram,
            max_relative_frequency=max_relative_frequency,
            min_absolute_frequency=min_absolute_frequency,
            max_features=max_features,
            sample=sample,
            full_text_col=full_text_col,
            id_col=id_col,
        )
        # Initialize topic model
        if model_type == 'NMF':
            topic_model = NonNegativeMatrixFactorization(corpus=corpus)
        elif model_type == 'LDA':
            topic_model = LatentDirichletAllocation(corpus=corpus)

        logger.info(f'Corpus size: {topic_model.corpus.size:,}')
        logger.info(f'Vocabulary size: {topic_model.corpus.vocabulary_size:,}')

        # Infer topics
        logger.info(f'Inferring {num_topics} topics')
        if model_type == 'NMF':
            topic_model.infer_topics(
                num_topics=num_topics,
                nmf_init=nmf_init,
                nmf_solver=nmf_solver,
                nmf_beta_loss=nmf_beta_loss,
                nmf_max_iter=nmf_max_iter,
                nmf_alpha=nmf_alpha,
                nmf_l1_ratio=nmf_l1_ratio,
                nmf_shuffle=nmf_shuffle,
                verbose=verbose,
                random_state=random_state,
            )
        elif model_type == 'LDA':
            topic_model.infer_topics(
                num_topics=num_topics,
                lda_algorithm=lda_algorithm,
                lda_alpha=lda_alpha,
                lda_eta=lda_eta,
                lda_learning_method=lda_learning_method,
                lda_n_jobs=lda_n_jobs,
                lda_n_iter=lda_n_iter,
                verbose=verbose,
                random_state=random_state,
            )

        # Save model on disk
        logger.info(f'Saving topic model: {topic_model_filepath}')
        ut.save_topic_model(topic_model, static_folder / topic_model_filepath)

    topic_cols_all = [' '.join(tw) for tw in topic_model.top_words_topics(num_words=top_words_description)]
    if rename_topics:
        rename = {tc: d for tc, d in zip(topic_cols_all, rename_topics)}
    else:
        rename = None

    # Get the top words for each topic for use around the site
    topic_description = [
        f"Topic {i:2d}: {rename_topics[i] + ' --- ' if rename_topics else None}{', '.join(tw)}" for i, tw in enumerate(
            topic_model.top_words_topics(num_words=top_words_description)
        )
    ]

    # Save the top words to CSV
    num_top_words_save = 20
    logger.info(f'Saving top {num_top_words_save} words CSV and XLSX')
    top_words_filename = f'{topic_model.corpus.name}_{topic_model.nb_topics}_topics_top_{num_top_words_save}_words'
    ut.save_top_words(num_top_words_save, topic_model, static_folder / data_folder / top_words_filename)

    # Get the vocabularly and split into sublists
    n_cols = 5
    words_per_col = int(ceil(topic_model.corpus.vocabulary_size / n_cols))
    split_vocabulary = [sublist for sublist in ut.chunks([(k, v) for k, v in topic_model.corpus.vocabulary.items()], words_per_col)]

    # Export topic cloud
    logger.info('Saving topic cloud')
    ut.save_topic_cloud(topic_model, static_folder / topic_cloud_folder / 'topic_cloud.json', top_words=top_words_cloud)

    # # Export per-topic author network using the most likely documents for each topic
    # logger.info('Saving author network details')
    # for topic_id in range(topic_model.nb_topics):
    #     ut.save_json_object(topic_model.corpus.collaboration_network(topic_model.documents_for_topic(topic_id)),
    #                         static_folder / author_network_folder / f'author_network{topic_id}.json')

    logger.info('Done.')

    # ##################################
    # Make plots for the main index page
    # ##################################

    logger.info('Creating plots...')

    # always create these images so they are up to date, and we have the paths based on the variables

    normalized = True
    thresh = 0.1
    freq = '1YS'
    ma_window = None
    savefig = True
    ncols = 7
    nchar_title = 30
    dpi = 72
    figformat = 'png'
    by_affil_list = [False, True]
    if merge_topics:
        merge_topics_list = [False, True]
    else:
        merge_topics_list = [False, False]

    viz = Visualization(topic_model, output_dir=static_folder / figs_folder)

    logger.info(f'Will save figures and figure data to: {viz.output_dir}')

    # count
    docs_over_time_count_line, docs_over_time_count_filepath = viz.plotly_docs_over_time(
        freq=freq,
        count=True,
        by_affil=True,
        ma_window=ma_window,
        output_type='div',
        savedata=True,
    )

    # percent
    docs_over_time_percent_line, docs_over_time_percent_filepath = viz.plotly_docs_over_time(
        freq=freq,
        count=False,
        by_affil=True,
        ma_window=ma_window,
        output_type='div',
        savedata=True,
    )

    # average topic loading
    topic_loading_barplot, topic_loading_filepath = viz.plotly_doc_topic_loading(
        rename=rename,
        normalized=normalized,
        n_words=top_words_description,
        output_type='div',
        savedata=True,
    )

    # topic_heatmap, topic_heatmap_filepath = viz.plotly_heatmap(
    #     rename=rename,
    #     normalized=normalized,
    #     n_words=top_words_description,
    #     annotate=True,
    #     annot_decimals=2,
    #     annot_fontsize=7,
    #     annot_fontcolor='black',
    #     output_type='div',
    #     savedata=False,
    # )

    topic_clustermap, topic_clustermap_filepath, topic_heatmap_filepath = viz.plotly_clustermap(
        rename=rename,
        normalized=normalized,
        n_words=top_words_description,
        annotate=True,
        annot_decimals=2,
        annot_fontsize=7,
        annot_fontcolor='black',
        output_type='div',
        savedata=True,
    )

    totc = []
    totp = []
    # totl = []
    for i, mt in enumerate(merge_topics_list):
        for ba in by_affil_list:
            if (not any(merge_topics_list)) and (i == 1):
                fig_topic_over_time_count = None
            else:
                _, _, fig_topic_over_time_count = viz.plot_topic_over_time_count(
                    rename=rename,
                    merge_topics=merge_topics if mt else None,
                    normalized=normalized,
                    thresh=thresh,
                    freq=freq,
                    n_words=top_words_description,
                    by_affil=ba,
                    ma_window=ma_window,
                    nchar_title=nchar_title,
                    ncols=ncols,
                    savefig=savefig,
                    dpi=dpi,
                    figformat=figformat,
                )
            totc.append(fig_topic_over_time_count)

            if (not any(merge_topics_list)) and (i == 1):
                fig_topic_over_time_percent = None
            else:
                _, _, fig_topic_over_time_percent = viz.plot_topic_over_time_percent(
                    rename=rename,
                    merge_topics=merge_topics if mt else None,
                    normalized=normalized,
                    thresh=thresh,
                    freq=freq,
                    n_words=top_words_description,
                    by_affil=ba,
                    ma_window=ma_window,
                    nchar_title=nchar_title,
                    ncols=ncols,
                    savefig=savefig,
                    dpi=dpi,
                    figformat=figformat,
                )
            totp.append(fig_topic_over_time_percent)

            # if (not any(merge_topics_list)) and (i == 1):
            #     fig_topic_over_time_loading = None
            # else:
            #     _, _, fig_topic_over_time_loading = viz.plot_topic_over_time_loading(
            #         rename=rename,
            #         merge_topics=merge_topics if mt else None,
            #         normalized=normalized,
            #         thresh=thresh,
            #         freq=freq,
            #         n_words=top_words_description,
            #         by_affil=ba,
            #         ma_window=ma_window,
            #         nchar_title=nchar_title,
            #         ncols=ncols,
            #         savefig=savefig,
            #         dpi=dpi,
            #         figformat=figformat,
            #     )
            # totl.append(fig_topic_over_time_loading)

    # _, _, fig_topic_topic_corr_heatmap = viz.plot_heatmap(
    #     rename=rename,
    #     normalized=normalized,
    #     fmt='.2f',
    #     annot_fontsize=12,
    #     n_words=top_words_description,
    #     savefig=savefig,
    #     dpi=dpi,
    #     figformat=figformat,
    # )

    _, fig_topic_topic_corr_clustermap = viz.plot_clustermap(
        rename=rename,
        normalized=normalized,
        fmt='.2f',
        annot_fontsize=12,
        n_words=top_words_description,
        savefig=savefig,
        dpi=dpi,
        figformat=figformat,
    )

    # # debug
    # fig_topic_over_time_count = ''
    # fig_topic_over_time_percent = ''
    # fig_topic_over_time_loading = ''
    # fig_topic_over_time_count_affil = ''
    # fig_topic_over_time_percent_affil = ''
    # fig_topic_over_time_loading_affil = ''
    # fig_topic_topic_corr_heatmap = ''
    # fig_topic_topic_corr_clustermap = ''

    logger.info('Done.')

    # ##################################
    # Print info
    # ##################################

    topic_model.print_topics(num_words=10)

    server = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

    # ##################################
    # Set up topic loading similarity app
    # ##################################

    external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
    ]

    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/topic_loading_similarity/',
        external_stylesheets=external_stylesheets,
    )

    app.title = 'Topic Loading Similarity'
    similarity_col = 'similarity'

    cols_sim = [
        similarity_col,
        topic_model.corpus._title_col,
        topic_model.corpus._dataset_col,
        topic_model.corpus._affiliation_col,
        topic_model.corpus._author_col,
        topic_model.corpus._date_col,
        id_col,
    ]
    cols_nosim = [c for c in cols_sim if c in topic_model.corpus.data_frame.columns]

    app.layout = html.Div([
        html.Div([
            html.Div(
                html.P('Drag or click the sliders to describe a topic loading vector. The most similar documents are displayed below.'),
                style={'float': 'left'},
            ),
            html.Div(
                html.A('Back to topic browser', id='back-to-main', href='../'),
                style={'float': 'right'},
            ),
        ]),
        html.Div(html.P('')),
        html.Div([
            html.Div([
                html.Div(
                    dcc.Slider(
                        id=f'slider-topic-{n}',
                        min=0.0,
                        max=1.0,
                        step=0.1,
                        value=0.0,  # starting value
                        updatemode='drag',
                    ),
                    style={
                        'width': '20%',
                        'display': 'inline-block',
                    },
                ),
                html.Div(
                    id=f'slider-output-container-{n}',
                    style={
                        'marginLeft': 10,
                        'marginRight': 5,
                        'font-size': 'small',
                        'display': 'inline-block',
                    },
                ),
                html.Div(
                    html.Label(
                        topic_description[n]
                    ),
                    style={
                        'font-weight': 'bold',
                        'font-size': 'small',
                        'width': '75%',
                        'display': 'inline-block',
                    },
                ),
            ]) for n in range(topic_model.nb_topics)],
            style={'width': '100%', 'display': 'inline-block'},
        ),
        html.Label('Number of documents to display'),
        html.Div(
            dcc.Dropdown(
                id='num-docs-dropdown',
                options=[
                    {'label': '10', 'value': 10},
                    {'label': '50', 'value': 50},
                    {'label': '100', 'value': 100},
                    {'label': '200', 'value': 200},
                    {'label': 'All', 'value': topic_model.corpus.size},
                ],
                value=10,
                placeholder='Select...',
            ),
            style={
                'width': '10%',
                'display': 'inline-block',
            },
        ),
        html.Div(
            html.A(
                html.Button('Export to CSV'),
                id='download-link',
                download=f'{corpus_name}_topic_loading_similarity.csv',
                href='',
                target='_blank',
            ),
            style={
                'display': 'inline-block',
                'float': 'right',
            },
        ),
        html.Div([
            dt.DataTable(
                id='doc-table',
                data=[],
                columns=[{"name": i, "id": i} for i in cols_sim],
                style_table={'overflowX': 'scroll'},
                style_cell={
                    'minWidth': '0px', 'maxWidth': '250px',
                    'whiteSpace': 'normal'
                },
                style_cell_conditional=[
                    {'if': {'column_id': similarity_col},
                        'width': '7%'},
                    {'if': {'column_id': topic_model.corpus._title_col},
                        'width': '39%'},
                    {'if': {'column_id': topic_model.corpus._dataset_col},
                        'width': '6%'},
                    {'if': {'column_id': topic_model.corpus._affiliation_col},
                        'width': '14%'},
                    {'if': {'column_id': topic_model.corpus._author_col},
                        'width': '12%'},
                    {'if': {'column_id': topic_model.corpus._date_col},
                        'width': '7%'},
                    {'if': {'column_id': id_col},
                        'width': '15%'},
                ],
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                css=[{
                    'selector': '.dash-cell div.dash-cell-value',
                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                }],
                editable=False,
                row_deletable=False,
                filter_action='native',
                sort_action='native',
                page_action='native',
                page_current=0,
                page_size=100,
                style_as_list_view=False,
            ),
        ]),
    ])

    for n in range(topic_model.nb_topics):
        @app.callback(
            Output(f'slider-output-container-{n}', 'children'),
            [Input(f'slider-topic-{n}', 'value')],
        )
        def update_output(slider_n_value):
            return f'{slider_n_value:.1f}'

    def filter_data(vector, num_docs=None, round_decimal=None):
        if not num_docs:
            num_docs = 10
        if not round_decimal:
            round_decimal = 4
        doc_ids_sims = topic_model.similar_documents(vector, num_docs=num_docs)
        doc_ids = [x[0] for x in doc_ids_sims]
        result = topic_model.corpus.data_frame.reindex(columns=cols_nosim, index=doc_ids)
        result[similarity_col] = [round(x[1], round_decimal) for x in doc_ids_sims]
        result[topic_model.corpus._date_col] = result[topic_model.corpus._date_col].dt.strftime('%Y-%m-%d')
        return result

    @app.callback(
        Output('doc-table', 'data'),
        [Input(f'slider-topic-{n}', 'value') for n in range(topic_model.nb_topics)] + [Input('num-docs-dropdown', 'value')],
    )
    def update_table(*args):
        vector = list(args[:-1])
        num_docs = args[-1]
        return filter_data(vector, num_docs).to_dict('records')

    @app.callback(
        Output('download-link', 'href'),
        [Input(f'slider-topic-{n}', 'value') for n in range(topic_model.nb_topics)] + [Input('num-docs-dropdown', 'value')],
    )
    def update_download_link(*args):
        vector = list(args[:-1])
        num_docs = args[-1]
        return 'data:text/csv;charset=utf-8,%EF%BB%BF' + urllib.parse.quote(
            filter_data(vector, num_docs).to_csv(index=False, encoding='utf-8')
        )

    # ##################################
    # Serve pages
    # ##################################

    @server.route('/')
    def index():
        return render_template(
            'index.html',
            topic_ids=topic_description,
            doc_ids=range(topic_model.corpus.size),
            method=type(topic_model).__name__,
            corpus_name=corpus_name,
            corpus_size=topic_model.corpus.size,
            vocabulary_size=topic_model.corpus.vocabulary_size,
            max_relative_frequency=max_relative_frequency,
            min_absolute_frequency=min_absolute_frequency,
            vectorization=vectorization,
            num_topics=num_topics,
            random_state=topic_model.random_state,
            top_words_csv=data_folder / f'{top_words_filename}.csv',
            top_words_xlsx=data_folder / f'{top_words_filename}.xlsx',
            docs_over_time_count_line=docs_over_time_count_line,
            docs_over_time_count_filepath=figs_folder / docs_over_time_count_filepath,
            docs_over_time_percent_line=docs_over_time_percent_line,
            docs_over_time_percent_filepath=figs_folder / docs_over_time_percent_filepath,
            topic_loading_barplot=topic_loading_barplot,
            topic_loading_filepath=figs_folder / topic_loading_filepath,
            # topic_heatmap=topic_heatmap,
            topic_clustermap=topic_clustermap,
            topic_clustermap_filepath=figs_folder / topic_clustermap_filepath,
            topic_heatmap_filepath=figs_folder / topic_heatmap_filepath,
            fig_topic_over_time_count=figs_folder / totc[0] if totc[0] else None,  # count, original topics, combined affiliations
            fig_topic_over_time_percent=figs_folder / totp[0] if totp[0] else None,  # percent, original topics, combined affiliations
            # fig_topic_over_time_loading=figs_folder / totl[0] if totl[0] else None,  # loading, original topics, combined affiliations
            fig_topic_over_time_count_affil=figs_folder / totc[1] if totc[1] else None,  # count, original topics, split affiliations
            fig_topic_over_time_percent_affil=figs_folder / totp[1] if totp[1] else None,  # percent, original topics, split affiliations
            # fig_topic_over_time_loading_affil=figs_folder / totl[1] if totl[1] else None,  # loading, original topics, split affiliations
            fig_topic_over_time_count_merged=figs_folder / totc[2] if totc[2] else None,  # count, merged topics, combined affiliations
            fig_topic_over_time_percent_merged=figs_folder / totp[2] if totp[2] else None,  # percent, merged topics, combined affiliations
            # fig_topic_over_time_loading_merged=figs_folder / totl[2] if totl[2] else None,  # loading, merged topics, combined affiliations
            fig_topic_over_time_count_affil_merged=figs_folder / totc[3] if totc[3] else None,  # count, merged topics, split affiliations
            fig_topic_over_time_percent_affil_merged=figs_folder / totp[3] if totp[3] else None,  # percent, merged topics, split affiliations
            # fig_topic_over_time_loading_affil_merged=figs_folder / totl[3] if totl[3] else None,  # loading, merged topics, split affiliations
            # fig_topic_topic_corr_heatmap=figs_folder / fig_topic_topic_corr_heatmap,
            fig_topic_topic_corr_clustermap=figs_folder / fig_topic_topic_corr_clustermap,
        )

    @server.route('/topic_cloud.html')
    def topic_cloud():
        return render_template(
            'topic_cloud.html',
            topic_ids=topic_description,
            doc_ids=range(topic_model.corpus.size),
            topic_cloud_filename=topic_cloud_folder / 'topic_cloud.json',
        )

    @server.route('/vocabulary.html')
    def vocabulary():
        return render_template(
            'vocabulary.html',
            topic_ids=topic_description,
            split_vocabulary=split_vocabulary,
            vocabulary_size=topic_model.corpus.vocabulary_size,
        )

    @server.route('/topic/<tid>.html')
    def topic_details(tid: str):
        tid = int(tid)
        # get the most likely documents per topic
        ids = topic_model.documents_for_topic(tid)
        # # get the top 100 documents per topic
        # ids = list(topic_model.top_topic_docs(topics=tid, top_n=100))[0][1]
        documents = []
        for i, document_id in enumerate(ids):
            documents.append(
                (
                    i + 1,
                    topic_model.corpus.title(document_id).title(),
                    ', '.join(topic_model.corpus.dataset(document_id)).title(),
                    ', '.join(topic_model.corpus.affiliation(document_id)).title(),
                    ', '.join(topic_model.corpus.author(document_id)).title(),
                    topic_model.corpus.date(document_id).strftime('%Y-%m-%d'),
                    topic_model.corpus.id(document_id),
                    document_id,
                ),
            )

        topic_word_weight_barplot, _ = viz.plotly_topic_word_weight(
            tid, normalized=True, n_words=20, output_type='div', savedata=False)
        topic_over_time_percent_line, _ = viz.plotly_topic_over_time(
            tid, count=False, output_type='div', savedata=False)
        topic_affiliation_count_barplot, _ = viz.plotly_topic_affiliation_count(
            tid, output_type='div', savedata=False)

        return render_template(
            'topic.html',
            topic_id=tid,
            description=f"{tid}{': ' + rename_topics[tid] if rename_topics else None}",
            frequency=round(topic_model.topic_frequency(tid) * 100, 2),
            documents=documents,
            topic_ids=topic_description,
            doc_ids=range(topic_model.corpus.size),
            topic_word_weight_barplot=topic_word_weight_barplot,
            topic_over_time_percent_line=topic_over_time_percent_line,
            topic_affiliation_count_barplot=topic_affiliation_count_barplot,
            # author_network_filename=author_network_folder / f'author_network{tid}.json',
        )

    @server.route('/document/<did>.html')
    def document_details(did: str):
        did = int(did)
        vector = topic_model.corpus.word_vector_for_document(did)
        word_list = []
        for a_word_id in range(len(vector)):
            word_list.append((topic_model.corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
        word_list = sorted(word_list, key=lambda x: x[1], reverse=True)
        documents = []
        for another_doc in topic_model.corpus.similar_documents(did, 5):
            documents.append(
                (
                    topic_model.corpus.title(another_doc[0]).title(),
                    ', '.join(topic_model.corpus.author(another_doc[0])).title(),
                    topic_model.corpus.date(another_doc[0]).strftime('%Y-%m-%d'),
                    ', '.join(topic_model.corpus.affiliation(another_doc[0])).title(),
                    ', '.join(topic_model.corpus.dataset(another_doc[0])).title(),
                    another_doc[0],
                    round(another_doc[1], 3),
                ),
            )

        doc_topic_loading_barplot, _ = viz.plotly_doc_topic_loading(
            did,
            rename=rename,
            normalized=True,
            n_words=top_words_description,
            output_type='div',
            savedata=False,
        )

        return render_template(
            'document.html',
            doc_id=did,
            words=word_list[:21],
            topic_ids=topic_description,
            doc_ids=range(topic_model.corpus.size),
            documents=documents,
            title=topic_model.corpus.title(did).title(),
            authors=', '.join(topic_model.corpus.author(did)).title(),
            year=topic_model.corpus.date(did).strftime('%Y-%m-%d'),
            short_content=topic_model.corpus.title(did).title(),
            affiliation=', '.join(topic_model.corpus.affiliation(did)).title(),
            dataset=', '.join(topic_model.corpus.dataset(did)).title(),
            id=topic_model.corpus.id(did),
            full_text=topic_model.corpus.full_text(did),
            doc_topic_loading_barplot=doc_topic_loading_barplot,
        )

    @server.route('/word/<wid>.html')
    def word_details(wid: str):
        wid = int(wid)
        documents = []
        for document_id in topic_model.corpus.docs_for_word(wid, sort=True):
            documents.append(
                (
                    topic_model.corpus.title(document_id).title(),
                    ', '.join(topic_model.corpus.author(document_id)).title(),
                    topic_model.corpus.date(document_id).strftime('%Y-%m-%d'),
                    ', '.join(topic_model.corpus.affiliation(document_id)).title(),
                    ', '.join(topic_model.corpus.dataset(document_id)).title(),
                    document_id,
                ),
            )

        word_topic_loading_barplot, _ = viz.plotly_word_topic_loading(
            wid,
            rename=rename,
            normalized=True,
            n_words=top_words_description,
            output_type='div',
            savedata=False,
        )

        return render_template(
            'word.html',
            word_id=wid,
            word=topic_model.corpus.word_for_id(wid),
            topic_ids=topic_description,
            doc_ids=range(topic_model.corpus.size),
            documents=documents,
            word_topic_loading_barplot=word_topic_loading_barplot,
        )

    @app.server.route('/favicon.ico')
    def favicon():
        return send_from_directory(
            static_folder / 'images', request.path[1:],
            mimetype='image/vnd.microsoft.icon')

    @server.route('/robots.txt')
    def robots_txt():
        return send_from_directory(static_folder, request.path[1:])

    # @server.url_defaults
    # def hashed_static_file(endpoint, values):
    #     """Flask: add static file's cache invalidator param (last modified time)
    #     to URLs generated by url_for(). Blueprints aware.
    #     """
    #     if 'static' == endpoint or endpoint.endswith('.static'):
    #         filename = values.get('filename')
    #         if filename:
    #             blueprint = request.blueprint
    #             if '.' in endpoint:  # blueprint
    #                 blueprint = endpoint.rsplit('.', 1)[0]

    #             static_folder = server.static_folder
    #             # use blueprint, but dont set `static_folder` option
    #             if blueprint and server.blueprints[blueprint].static_folder:
    #                 static_folder = server.blueprints[blueprint].static_folder

    #             fp = Path(static_folder, filename)
    #             if fp.exists():
    #                 values['_'] = int(fp.stat().st_mtime)

    return app


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config_filepath = args.config_filepath
    config = get_config(config_filepath)

    config_section = 'webserver'
    port = config[config_section].getint('port', 5000)
    debug = config[config_section].getboolean('debug', False)

    app = main(config[config_section])

    # Access the browser at http://localhost:5000/
    # app.run_server(debug=debug, host='localhost', port=port)
    app.run_server(debug=debug, host='0.0.0.0', port=port)
