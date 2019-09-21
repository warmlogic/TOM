# TOM

This is a heavily overhauled version of [this library](http://mediamining.univ-lyon2.fr/people/guille/tom.php) ([GitHub repo](https://github.com/AdrienGuille/TOM)). New features include:
- configuration files
- ability to set topic model hyperparameters
- major computational speedups
- additional assessment metrics for choosing an appropriate number of topics
- normalized topic loadings
- new charts
- interactive charts
- access to raw data
- a topic loading similarity browser app

TOM (TOpic Modeling) is a Python 3 library for topic modeling and browsing, licensed under the MIT license. Its objective is to allow for an efficient analysis of a text corpus from start to finish, via the discovery of latent topics. To this end, TOM features functions for preparing and vectorizing a text corpus. It also offers a common interface for two topic models (namely LDA using either variational inference or Gibbs sampling, and NMF using alternating least-square with a projected gradient method), and implements three state-of-the-art methods for estimating the optimal number of topics to model a corpus. What is more, TOM constructs an interactive Web-based browser that makes it easy to explore a topic model and the related corpus.

## Features

### Vector space modeling

- Feature selection based on word frequency
- Weighting
  - tf
  - tf-idf

### Topic modeling

- Latent Dirichlet Allocation
  - Standard variational Bayesian inference ([Latent Dirichlet Allocation. Blei et al., 2003](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf))
  - Online variational Bayesian inference ([Online learning for Latent Dirichlet Allocation. Hoffman et al., 2010](http://papers.nips.cc/paper/3902-online-learning-for-latentdirichlet-allocation!))
  - Collapsed Gibbs sampling ([Finding scientific topics. Griffiths & Steyvers, 2004](https://doi.org/10.1073/pnas.0307752101))
- Non-negative Matrix Factorization (NMF)
  - Alternating least-square with a projected gradient method ([Projected gradient methods for non-negative matrix factorization. Lin, 2007](https://doi.org/10.1162/neco.2007.19.10.2756))

### Estimating the optimal number of topics

- Stability analysis ([How Many Topics? Stability Analysis for Topic Models. Greene et al, 2014](https://arxiv.org/abs/1404.4606))
- Spectral analysis ([On finding the natural number of topics with Latent Dirichlet Allocation: Some observations. Arun et al., 2010](https://doi.org/10.1007/978-3-642-13657-3_43))
- Consensus-based analysis ([Metagenes and molecular pattern discovery using matrix factorization. Brunet et al., 2004](https://dx.doi.org/10.1073%2Fpnas.0308531101))
- Word2Vec-based coherence metric
- Perplexity (LDA only; Measures perplexity for LDA as [computed by scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity).)

## Installation

- We recommend you install [Anaconda](https://www.continuum.io) (Python 3) which will automatically install most of the required dependencies (i.e., pandas, numpy, scipy, scikit-learn, matplotlib, flask).
- Then install the following libraries
  - [lda](https://github.com/ariddell/lda/) (`pip install lda`)
  - [gensim](https://radimrehurek.com/gensim/) (`pip install gensim`)
  - [smart_open](https://github.com/RaRe-Technologies/smart_open) (`pip install smart_open`)
  - [Plotly and Dash](https://dash.plot.ly) (`pip install plotly dash dash-daq`)

## Usage

We provide two programs:

- `assess_topics.py`: Produces artifacts used for estimating the optimal number of topics
- `build_topic_model_browser.py`: Runs a local web server and generates a web browser application for exploring the corpus and topic model

Both of these run based on your configuration file. Copy the `config_template.ini` file to `config.ini` and set it up as desired.

### Load and prepare a textual corpus

A corpus is a TSV (tab separated values) file describing documents. This is formatted as one document per line, with the following columns:

- `id`: an incrementing integer used for accessing documents by index
- `access_num`: a unique identifier string
- `affiliation`: for grouping documents within a dataset
- `dataset`: used when combining documents from various sources
- `title`
- `author`
- `date`: preferably formatted as `YYYY-MM-DD`
- `orig_text`: the original text of the document
- `text`: the text on which to train the topic model, which may be preprocessed in various ways

```tsv
id	access_num	affiliation	dataset	title	author	date	orig_text	text
1	doi123	journal1	dataset1	Document 1's title	Author 1	2019-01-01	Full content of document 1.	full content document 1
2	doi456	journal2	dataset1	Document 2's title	Author 2	2019-05-01	Full content of document 2.	full content document 2
etc.
```

The following code snippet shows how to load a corpus vectorize them using tf-idf with unigrams.

```python
corpus = Corpus(
    source_filepath='input/raw_corpus.tsv',
    vectorization='tfidf',
    n_gram=1,
    max_relative_frequency=0.8,
    min_absolute_frequency=4,
)
print(f'Corpus size: {corpus.size:,}')
print(f'Vocabulary size: {corpus.vocabulary_size:,}')
print('Vector representation of document 0:\n', corpus.word_vector_for_document(doc_id=0))
```

### Instantiate a topic model and infer topics

It is possible to instantiate a NMF or LDA object then infer topics.

```python
from tom_lib.structure.corpus import Corpus
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.visualization.visualization import Visualization
```

NMF (use `vectorization='tfidf'`):

```python
topic_model = NonNegativeMatrixFactorization(corpus)
topic_model.infer_topics(num_topics=15)
```

LDA (using either the standard variational Bayesian inference or Gibbs sampling; use `vectorization='tf'`):

```python
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='variational')
```

```python
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='gibbs')
```

### Instantiate a topic model and estimate the optimal number of topics

Here we instantiate a NMF object, then generate plots with the three metrics for estimating the optimal number of topics.

```python
topic_model = NonNegativeMatrixFactorization(corpus)
viz = Visualization(topic_model)
viz.plot_greene_metric(
    min_num_topics=5,
    max_num_topics=50,
    step=1,
    tao=10,
    top_n_words=10,
)
viz.plot_arun_metric(
    min_num_topics=5,
    max_num_topics=50,
    step=1,
    iterations=10,
)
viz.plot_brunet_metric(
    min_num_topics=5,
    max_num_topics=50,
    step=1,
    iterations=10,
)
viz.plot_coherence_w2v_metric(
    min_num_topics=5,
    max_num_topics=50,
    step=1,
    top_n_words=10,
)
# # LDA only
# viz.plot_perplexity_metric(
#     min_num_topics=5,
#     max_num_topics=50,
#     step=1,
# )
```

### Save/load a topic model

To allow reusing previously learned topics models, TOM can save them on disk, as shown below.

```python
import tom_lib.utils as ut
ut.save_topic_model(topic_model, 'output/NMF_15topics.pickle')
topic_model = ut.load_topic_model('output/NMF_15topics.pickle')
```

### Print information about a topic model

This code excerpt illustrates how one can manipulate a topic model, e.g. get the topic distribution for a document or the word distribution for a topic.

```python
print('\nTopics:')
topic_model.print_topics(num_words=10)
print('\nTopic distribution for document 0:',
      topic_model.topic_distribution_for_document(0))
print('\nMost likely topic for document 0:',
      topic_model.most_likely_topic_for_document(0))
print('\nFrequency of topics:',
      topic_model.topics_frequency())
print('\nTop 10 most relevant words for topic 2:',
      topic_model.top_words(2, 10))
```

## Running the web server

TODO: Add documentation based on these links

- https://medium.com/@akshatuppal/deploying-multiple-python-flask-applications-on-ec2-instance-using-nginx-and-gunicorn-32651fb3d064
- https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https
- https://stackoverflow.com/questions/7023052/configure-flask-dev-server-to-be-visible-across-the-network
- https://flask.palletsprojects.com/en/1.1.x/quickstart/
