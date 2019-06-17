# coding: utf-8
import configparser
from pathlib import Path, PurePath
import os
import tom_lib.utils as ut
from flask import Flask, render_template, request, send_from_directory
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization
import logging
logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)

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
max_relative_frequency = config[webserver_section].getfloat('max_relative_frequency', 0.8)
min_absolute_frequency = config[webserver_section].getint('min_absolute_frequency', 5)
num_topics = config[webserver_section].getint('num_topics', 15)
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
top_words_description = config[webserver_section].getint('top_words_description', 10)
top_words_cloud = config[webserver_section].getint('top_words_cloud', 5)
model_type = config[webserver_section].get('model_type', 'NMF')
nmf_beta_loss = config[webserver_section].get('nmf_beta_loss', 'frobenius')
lda_algorithm = config[webserver_section].get('lda_algorithm', 'variational')
load_if_existing_model = config[webserver_section].getboolean('load_if_existing_model', True)

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
word_distribution_folder = data_folder / 'word_distribution'
frequency_folder = data_folder / 'frequency'
affiliation_repartition_folder = data_folder / 'affiliation_repartition'
# author_network_folder = data_folder / 'author_network'
topic_distribution_d_folder = data_folder / 'topic_distribution_d'
topic_distribution_w_folder = data_folder / 'topic_distribution_w'
figs_folder = data_folder / 'figs'

full_text_col = 'orig_text'
id_col = 'access_num'

app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

# ##################################
# Load or train model
# ##################################

if load_if_existing_model and (static_folder / topic_model_filepath).exists():
    # Load model from disk:
    logger.info(f'Loading topic model: {static_folder / topic_model_filepath}')
    topic_model = ut.load_topic_model(static_folder / topic_model_filepath)

    logger.info(f'Corpus size: {topic_model.corpus.size:,}')
    logger.info(f'Vocabulary size: {len(topic_model.corpus.vocabulary):,}')
else:
    # Clean the topic model directory
    if (static_folder / tm_folder).exists():
        ut.delete_folder(static_folder / tm_folder)
    (static_folder / tm_folder).mkdir(parents=True, exist_ok=True)

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
                    full_text_col=full_text_col,
                    id_col=id_col,
                    )
    # Initialize topic model
    if model_type == 'NMF':
        topic_model = NonNegativeMatrixFactorization(corpus=corpus)
    elif model_type == 'LDA':
        topic_model = LatentDirichletAllocation(corpus=corpus)

    logger.info(f'Corpus size: {topic_model.corpus.size:,}')
    logger.info(f'Vocabulary size: {len(topic_model.corpus.vocabulary):,}')

    # Infer topics
    logger.info('Inferring topics')
    if model_type == 'NMF':
        topic_model.infer_topics(num_topics=num_topics, beta_loss=nmf_beta_loss)
    elif model_type == 'LDA':
        topic_model.infer_topics(num_topics=num_topics, algorithm=lda_algorithm)

    topic_description = []
    for i in range(topic_model.nb_topics):
        description = [weighted_word[0] for weighted_word in topic_model.top_words(i, top_words_description)]
        topic_description.append(f"Topic {i:2d}: {', '.join(description)}")

    # Save model on disk
    logger.info(f'Saving topic model: {topic_model_filepath}')
    ut.save_topic_model(topic_model, static_folder / topic_model_filepath)

    # Export topic cloud
    logger.info('Saving topic cloud')
    ut.save_topic_cloud(topic_model, static_folder / topic_cloud_folder / 'topic_cloud.json', top_words=top_words_cloud)

    # Export details about topics
    logger.info('Saving topic details')
    for topic_id in range(topic_model.nb_topics):
        ut.save_word_distribution(topic_model.top_words(topic_id, 20),
                                  static_folder / word_distribution_folder / f'word_distribution{topic_id}.tsv')

        ut.save_affiliation_repartition(topic_model.affiliation_repartition(topic_id),
                                        static_folder / affiliation_repartition_folder / f'affiliation_repartition{topic_id}.tsv')

        min_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].dt.year.min()
        max_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].dt.year.max()

        evolution = []
        for i in range(min_year, max_year + 1):
            evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
        ut.save_topic_evolution(evolution, static_folder / frequency_folder / f'frequency{topic_id}.tsv')

    # Export details about documents
    logger.info('Saving document details')
    for doc_id in range(topic_model.corpus.size):
        ut.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                   static_folder / topic_distribution_d_folder / f'topic_distribution_d{doc_id}.tsv',
                                   topic_description,
                                   )

    # Export details about words
    logger.info('Saving word details')
    for word_id in range(len(topic_model.corpus.vocabulary)):
        ut.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                   static_folder / topic_distribution_w_folder / f'topic_distribution_w{word_id}.tsv',
                                   topic_description,
                                   )

    # # Export per-topic author network using the most likely documents for each topic
    # logger.info('Saving author network details')
    # for topic_id in range(topic_model.nb_topics):
    #     ut.save_json_object(topic_model.corpus.collaboration_network(topic_model.documents_for_topic(topic_id)),
    #                         static_folder / author_network_folder / f'author_network{topic_id}.json')

logger.info('Done.')

# ##################################
# Make plots
# ##################################

logger.info('Creating plots...')

# always create these images so they are up to date, and we have the paths based on the variables

normalized = True
thresh = 0.1
freq = '1Y'
by_affil = False
ma_window = None
savefig = True
ncols = 7
nchar_title = 30
dpi = 72
figformat = 'png'

viz = Visualization(topic_model, output_dir=static_folder / figs_folder)

fig, ax, fig_docs_over_time_count = viz.plot_docs_over_time_count(
    freq=freq,
    by_affil=True,
    ma_window=ma_window,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

fig, ax, fig_docs_over_time_percent = viz.plot_docs_over_time_percent_affil(
    freq=freq,
    ma_window=ma_window,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

fig, ax, fig_topic_barplot = viz.plot_topic_loading_barplot(
    normalized=normalized,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

# fig, ax, fig_topic_heatmap = viz.plot_heatmap(
#     normalized=normalized,
#     savefig=savefig,
#     dpi=dpi,
#     figformat=figformat,
# )

g, fig_topic_clustermap = viz.plot_clustermap(
    normalized=normalized,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

fig, ax, fig_topic_over_time_count = viz.plot_topic_over_time_count(
    normalized=normalized,
    thresh=thresh,
    freq=freq,
    by_affil=by_affil,
    ma_window=ma_window,
    nchar_title=nchar_title,
    ncols=ncols,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

fig, ax, fig_topic_over_time_percent = viz.plot_topic_over_time_percent(
    normalized=normalized,
    thresh=thresh,
    freq=freq,
    by_affil=by_affil,
    ma_window=ma_window,
    nchar_title=nchar_title,
    ncols=ncols,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

fig, ax, fig_topic_over_time_loading = viz.plot_topic_over_time_loading(
    normalized=normalized,
    thresh=thresh,
    freq=freq,
    by_affil=by_affil,
    ma_window=ma_window,
    nchar_title=nchar_title,
    ncols=ncols,
    savefig=savefig,
    dpi=dpi,
    figformat=figformat,
)

# # debug
# fig_docs_over_time_count = None
# fig_docs_over_time_percent = None
# fig_topic_barplot = None
# # fig_topic_heatmap = None
# fig_topic_clustermap = None
# fig_topic_over_time_count = None
# fig_topic_over_time_percent = None
# fig_topic_over_time_loading = None

logger.info('Done.')

# ##################################
# Print info and serve pages
# ##################################

topic_model.print_topics(num_words=10)

topic_description = []
for i in range(topic_model.nb_topics):
    description = [weighted_word[0] for weighted_word in topic_model.top_words(i, top_words_description)]
    topic_description.append(f"Topic {i:2d}: {', '.join(description)}")


@app.route('/')
def index():
    return render_template(
        'index.html',
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        method=type(topic_model).__name__,
        corpus_size=topic_model.corpus.size,
        vocabulary_size=len(topic_model.corpus.vocabulary),
        max_relative_frequency=max_relative_frequency,
        min_absolute_frequency=min_absolute_frequency,
        vectorization=vectorization,
        num_topics=num_topics,
        fig_docs_over_time_count=figs_folder / fig_docs_over_time_count,
        fig_docs_over_time_percent=figs_folder / fig_docs_over_time_percent,
        fig_topic_barplot=figs_folder / fig_topic_barplot,
        # fig_topic_heatmap=figs_folder / fig_topic_heatmap,
        fig_topic_clustermap=figs_folder / fig_topic_clustermap,
        fig_topic_over_time_count=figs_folder / fig_topic_over_time_count,
        fig_topic_over_time_percent=figs_folder / fig_topic_over_time_percent,
        fig_topic_over_time_loading=figs_folder / fig_topic_over_time_loading,
    )


@app.route('/topic_cloud.html')
def topic_cloud():
    return render_template(
        'topic_cloud.html',
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        topic_cloud_filename=topic_cloud_folder / f'topic_cloud.json',
    )


@app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(topic_model.corpus.vocabulary)):
        word_list.append((i, topic_model.corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(topic_model.corpus.vocabulary) / 5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j * words_per_column, (j + 1) * words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template(
        'vocabulary.html',
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        splitted_vocabulary=splitted_vocabulary,
        vocabulary_size=len(word_list),
    )


@app.route('/topic/<tid>.html')
def topic_details(tid):
    # get the most likely documents per topic
    ids = topic_model.documents_for_topic(int(tid))
    # # get the top 100 documents per topic
    # ids = list(topic_model.top_topic_docs(topics=int(tid), top_n=100))[0][1]
    documents = []
    for i, document_id in enumerate(ids):
        documents.append((i + 1, topic_model.corpus.title(document_id).title(),
                          ', '.join(topic_model.corpus.affiliation(document_id)).title(),
                          ', '.join(topic_model.corpus.author(document_id)).title(),
                          topic_model.corpus.date(document_id), document_id))
    return render_template(
        'topic.html',
        topic_id=tid,
        frequency=round(topic_model.topic_frequency(int(tid)) * 100, 2),
        documents=documents,
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        word_distribution_filename=word_distribution_folder / f'word_distribution{tid}.tsv',
        frequency_filename=frequency_folder / f'frequency{tid}.tsv',
        affiliation_repartition_filename=affiliation_repartition_folder / f'affiliation_repartition{tid}.tsv',
        # author_network_filename=author_network_folder / f'author_network{tid}.json',
    )


@app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((topic_model.corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in topic_model.corpus.similar_documents(int(did), 5):
        documents.append(
            (topic_model.corpus.title(another_doc[0]).title(),
             ', '.join(topic_model.corpus.author(another_doc[0])).title(),
             topic_model.corpus.date(another_doc[0]),
             ', '.join(topic_model.corpus.affiliation(another_doc[0])).title(),
             another_doc[0],
             round(another_doc[1], 3),
             ),
        )
    return render_template(
        'document.html',
        doc_id=did,
        words=word_list[:21],
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        documents=documents,
        title=topic_model.corpus.title(int(did)).title(),
        authors=', '.join(topic_model.corpus.author(int(did))).title(),
        year=topic_model.corpus.date(int(did)),
        short_content=topic_model.corpus.title(int(did)).title(),
        affiliation=', '.join(topic_model.corpus.affiliation(int(did))).title(),
        id=topic_model.corpus.id(int(did)),
        full_text=topic_model.corpus.full_text(int(did)),
        topic_distribution_d_filename=topic_distribution_d_folder / f'topic_distribution_d{did}.tsv',
        doc_topic_loading_barplot=viz.plotly_doc_topic_loading(int(did), normalized=True, n_words=5, output_type='div'),
    )


@app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in topic_model.corpus.docs_for_word(int(wid), sort=True):
        documents.append(
            (topic_model.corpus.title(document_id).title(),
             ', '.join(topic_model.corpus.author(document_id)).title(),
             topic_model.corpus.date(document_id),
             ', '.join(topic_model.corpus.affiliation(document_id)).title(),
             document_id,
             ),
        )
    return render_template(
        'word.html',
        word_id=wid,
        word=topic_model.corpus.word_for_id(int(wid)),
        topic_ids=topic_description,
        doc_ids=range(topic_model.corpus.size),
        documents=documents,
        topic_distribution_w_filename=topic_distribution_w_folder / f'topic_distribution_w{wid}.tsv',
    )


@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(PurePath(app.root_path, 'static'), request.path[1:])


@app.url_defaults
def hashed_static_file(endpoint, values):
    '''Flask: add static file's cache invalidator param (last modified time)
    to URLs generated by url_for(). Blueprints aware.
    '''
    if 'static' == endpoint or endpoint.endswith('.static'):
        filename = values.get('filename')
        if filename:
            blueprint = request.blueprint
            if '.' in endpoint:  # blueprint
                blueprint = endpoint.rsplit('.', 1)[0]

            static_folder = app.static_folder
            # use blueprint, but dont set `static_folder` option
            if blueprint and app.blueprints[blueprint].static_folder:
                static_folder = app.blueprints[blueprint].static_folder

            fp = Path(static_folder, filename)
            if fp.exists():
                values['_'] = int(fp.stat().st_mtime)


if __name__ == '__main__':
    # Access the browser at http://localhost:2016/
    # app.run(debug=True, host='localhost', port=2016)
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=False, host='0.0.0.0', port=5000)
