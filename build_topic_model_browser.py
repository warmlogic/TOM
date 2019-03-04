# coding: utf-8
import configparser
from pathlib import Path
import os
import shutil
import tom_lib.utils as utils
from flask import Flask, render_template
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization
from tom_lib.structure.corpus import Corpus

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

config_filepath = 'config.ini'
config = configparser.ConfigParser(allow_no_value=True)
try:
    config.read(config_filepath)
except OSError as e:
    print(f'Config file {config_filepath} not found. Did you set it up?')

# Read parameters
data_dir = config['tom'].get('data_dir', '', vars=os.environ)
if not data_dir:
    data_dir = '.'
data_dir = Path(data_dir)
docs_filename = config['tom'].get('docs_filename', '')
if not docs_filename:
    raise ValueError(f'docs_filename not specified in {config_filepath}')

source_filepath = data_dir / docs_filename
# Ensure data exists
if not source_filepath.exists():
    raise OSError(f'Documents file does not exist: {source_filepath}')

language = config['tom'].get('language', '')
if not language:
    language = None
max_relative_frequency = config['tom'].getfloat('max_relative_frequency', 0.8)
min_absolute_frequency = config['tom'].getint('min_absolute_frequency', 5)
num_topics = config['tom'].getint('', 15)
vectorization = config['tom'].get('vectorization', 'tfidf')
n_gram = config['tom'].getint('n_gram', 1)
max_features = config['tom'].getint('max_features', 2000)
sample = config['tom'].getfloat('sample', 1.0)
top_words_description = config['tom'].getint('top_words_description', 5)
top_words_cloud = config['tom'].getint('top_words_cloud', 5)

# Flask Web server
app = Flask(__name__, static_folder='browser/static', template_folder='browser/templates')

# Load corpus
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
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))

# Infer topics
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=num_topics)
topic_model.print_topics(num_words=10)

topic_description = []
for i in range(topic_model.nb_topics):
    description = [weighted_word[0] for weighted_word in topic_model.top_words(i, top_words_description)]
    topic_description.append(f"Topic {i:2d}: {', '.join(description)}")

# Clean the data directory
if os.path.exists('browser/static/data'):
    shutil.rmtree('browser/static/data')
os.makedirs('browser/static/data')

# Export topic cloud
utils.save_topic_cloud(topic_model, 'browser/static/data/topic_cloud.json', top_words=top_words_cloud)

# Export details about topics
for topic_id in range(topic_model.nb_topics):
    utils.save_word_distribution(topic_model.top_words(topic_id, 20),
                                 f'browser/static/data/word_distribution{topic_id}.tsv')
    utils.save_affiliation_repartition(topic_model.affiliation_repartition(topic_id),
                                       f'browser/static/data/affiliation_repartition{topic_id}.tsv')

    min_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].min()
    max_year = topic_model.corpus.data_frame[topic_model.corpus._date_col].max()

    evolution = []
    for i in range(min_year, max_year + 1):
        evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, f'browser/static/data/frequency{topic_id}.tsv')

# Export details about documents
for doc_id in range(topic_model.corpus.size):
    utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                  f'browser/static/data/topic_distribution_d{doc_id}.tsv',
                                  )

# Export details about words
for word_id in range(len(topic_model.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                  f'browser/static/data/topic_distribution_w{word_id}.tsv',
                                  )

# # Associate documents with topics
# topic_associations = topic_model.documents_per_topic()

# # Export per-topic author network
# for topic_id in range(topic_model.nb_topics):
#     utils.save_json_object(corpus.collaboration_network(topic_associations[topic_id]),
#                            f'browser/static/data/author_network{topic_id}.json')


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


if __name__ == '__main__':
    # Access the browser at http://localhost:2016/
    # app.run(debug=True, host='localhost', port=2016)
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=False, host='0.0.0.0', port=5000)
