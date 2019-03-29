# coding: utf-8
import configparser
from pathlib import Path, PurePath
import os
import shutil
import tom_lib.utils as utils
from flask import Flask, render_template, request, send_from_directory
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus
import logging
logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

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
top_words_description = config[webserver_section].getint('top_words_description', 5)
top_words_cloud = config[webserver_section].getint('top_words_cloud', 5)
model_type = config[webserver_section].get('model_type', 'NMF')
nmf_beta_loss = config[webserver_section].get('nmf_beta_loss', 'frobenius')
lda_algorithm = config[webserver_section].get('lda_algorithm', 'variational')

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
static_folder = 'browser/static'
template_folder = 'browser/templates'
data_folder = static_folder / f'data/{num_topics}_topics'

app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

# Clean the data directory
if os.path.exists(data_folder):
    shutil.rmtree(data_folder)
os.makedirs(data_folder)

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

# Infer topics
logger.info('Inferring topics')
if model_type == 'NMF':
    topic_model.infer_topics(num_topics=num_topics, beta_loss=nmf_beta_loss)
elif model_type == 'LDA':
    topic_model.infer_topics(num_topics=num_topics, algorithm=lda_algorithm)

topic_model.print_topics(num_words=10)

topic_description = []
for i in range(topic_model.nb_topics):
    description = [weighted_word[0] for weighted_word in topic_model.top_words(i, top_words_description)]
    topic_description.append(f"Topic {i:2d}: {', '.join(description)}")

# Export topic cloud
logger.info('Saving topic cloud')
utils.save_topic_cloud(topic_model, f'{data_folder}/topic_cloud.json', top_words=top_words_cloud)

# Export details about topics
logger.info('Saving topic details')
for topic_id in range(topic_model.nb_topics):
    utils.save_word_distribution(topic_model.top_words(topic_id, 20),
                                 f'{data_folder}/word_distribution{topic_id}.tsv')
    utils.save_affiliation_repartition(topic_model.affiliation_repartition(topic_id),
                                       f'{data_folder}/affiliation_repartition{topic_id}.tsv')

    min_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].min()
    max_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].max()

    evolution = []
    for i in range(min_year, max_year + 1):
        evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, f'{data_folder}/frequency{topic_id}.tsv')

# Export details about documents
logger.info('Saving document details')
for doc_id in range(topic_model.corpus.size):
    utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                  f'{data_folder}/topic_distribution_d{doc_id}.tsv',
                                  topic_description,
                                  )

# Export details about words
logger.info('Saving word details')
for word_id in range(len(topic_model.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                  f'{data_folder}/topic_distribution_w{word_id}.tsv',
                                  topic_description,
                                  )

# # Associate documents with topics
# topic_associations = topic_model.documents_per_topic()

# # Export per-topic author network
# logger.info('Saving author network details')
# for topic_id in range(topic_model.nb_topics):
#     utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
#                            f'{data_folder}/author_network{topic_id}.json')


@app.route('/')
def index():
    return render_template('index.html',
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size),
                           method=type(topic_model).__name__,
                           corpus_size=corpus.size,
                           vocabulary_size=len(corpus.vocabulary),
                           max_relative_frequency=max_relative_frequency,
                           min_absolute_frequency=min_absolute_frequency,
                           vectorization=vectorization,
                           num_topics=num_topics)


@app.route('/topic_cloud.html')
def topic_cloud():
    return render_template('topic_cloud.html',
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size))


@app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary) / 5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j * words_per_column, (j + 1) * words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary.html',
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic/<tid>.html')
def topic_details(tid):
    # ids = topic_associations[int(tid)]
    ids = list(topic_model.top_topic_docs(topics=int(tid), top_n=100))[0][1]
    documents = []
    for i, document_id in enumerate(ids):
        documents.append((i + 1, corpus.title(document_id).title(),
                          ', '.join(corpus.affiliation(document_id)).title(),
                          ', '.join(corpus.author(document_id)).title(),
                          corpus.date(document_id), document_id))
    return render_template('topic.html',
                           topic_id=tid,
                           frequency=round(topic_model.topic_frequency(int(tid)) * 100, 2),
                           documents=documents,
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size))


@app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    documents = []
    for another_doc in corpus.similar_documents(int(did), 5):
        documents.append((corpus.title(another_doc[0]).title(),
                          ', '.join(corpus.author(another_doc[0])).title(),
                          corpus.date(another_doc[0]), another_doc[0], round(another_doc[1], 3)))
    return render_template('document.html',
                           doc_id=did,
                           words=word_list[:21],
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size),
                           documents=documents,
                           title=corpus.title(int(did)).title(),
                           authors=', '.join(corpus.author(int(did))).title(),
                           year=corpus.date(int(did)),
                           short_content=corpus.title(int(did)).title(),
                           affiliation=', '.join(corpus.affiliation(int(did))).title(),
                           full_text=corpus.full_text(int(did)),
                           )


@app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.title(document_id).title(),
                          ', '.join(corpus.author(document_id)).title(),
                          corpus.date(document_id), document_id))
    return render_template('word.html',
                           word_id=wid,
                           word=topic_model.corpus.word_for_id(int(wid)),
                           topic_ids=topic_description,
                           doc_ids=range(corpus.size),
                           documents=documents)


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
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=False, host='0.0.0.0', port=5000)
