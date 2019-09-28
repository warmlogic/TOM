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

TOM (TOpic Modeling) is a Python 3 library for topic modeling and browsing, licensed under the MIT license. Its objective is to allow for an efficient analysis of a text corpus from start to finish, via the discovery of latent topics. To this end, TOM features functions for preparing and vectorizing a text corpus, though you may want to perform additional preprocessing steps on the corpus before topic modeling. It also offers a common interface for two topic models (LDA using either variational inference or Gibbs sampling, and NMF using alternating least squares with a projected gradient method), and implements five state-of-the-art methods for estimating the optimal number of topics to model a corpus. TOM constructs an interactive web browser-based application that makes it easy to explore a topic model and the related corpus.

## Features

### Vector space modeling

- Feature selection based on word frequency
  - Via the parameters `max_relative_frequency`, `min_absolute_frequency`, and `max_features`
- Weighting
  - [tf-idf](https://en.wikipedia.org/wiki/Tf–idf) (for NMF)
  - tf (for LDA)

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

1. In a terminal, `cd` to the `TOM` directory
1. Run the following command to install [Miniconda (Python 3)](https://conda.io/miniconda.html) and install the required libraries in the `base` conda environment:

```bash
$ ./python_env_setup.sh
```

List of the installed libraries:

- [Plotly and Dash](https://dash.plot.ly)
- [gensim](https://radimrehurek.com/gensim/)
- [lda](https://github.com/ariddell/lda/)
- [matplotlib](https://matplotlib.org)
- [networkx](https://networkx.github.io)
- [nltk](https://www.nltk.org)
- [numpy](https://numpy.org)
- [openpyxl](https://openpyxl.readthedocs.io)
- [pandas](https://pandas.pydata.org)
- [scikit-learn](https://scikit-learn.org/)
- [scipy](https://www.scipy.org)
- [seaborn](https://seaborn.pydata.org)
- [smart_open](https://github.com/RaRe-Technologies/smart_open)

## Provided scripts

The provided scripts use the parameters defined in the configuration file. Copy the `config_template.ini` file to `config.ini` and set it up as desired.

In order of importance, the scripts are:

1. `assess_topics.py`: Produce artifacts used for estimating the optimal number of topics
1. `build_topic_model_browser.py`: Run a local web server and generate a web browser-based application for exploring the topic model and corpus
1. `infer_topics.py`: Simply train and save topic models for a range of numbers of topics

Run a script in the terminal using the following command structure:
```bash
$ python <script name> --config_filepath=<config file name>
```

## Expected corpus format

A corpus is a TSV (tab separated values) file describing documents. This is formatted as one document per line, with the following columns:

- `id`: a unique identifier
- `affiliation`: for grouping documents within a dataset
- `dataset`: used when combining documents from various sources
- `title`
- `author`
- `date`: preferably formatted as `YYYY-MM-DD`
- `text`: the text on which to train the topic model, which may be preprocessed in various ways
- `orig_text`: the original text of the document (optional; if absent, will use `text` column)

```
id	affiliation	dataset	title	author	date	text	orig_text
doc1	journal1	dataset1	Document 1's title	Author 1	2019-01-01	full content document 1	Full content of document 1.
doc2	journal2	dataset1	Document 2's title	Author 2	2019-05-01	full content document 2	Full content of document 2.
etc.
```

## Interactive usage

The following code snippets get run in the provided scripts. They are shown below to demonstrate how to interact with them. You'll need to import the required classes as follows:

```python
from tom_lib.structure.corpus import Corpus
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.visualization.visualization import Visualization
```

### Load and prepare a corpus

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

NMF (use `vectorization='tfidf'` when creating the corpus):

```python
topic_model = NonNegativeMatrixFactorization(corpus)
topic_model.infer_topics(num_topics=15)
```

LDA (using either the standard variational Bayesian inference or Gibbs sampling; use `vectorization='tf'` when creating the corpus):

```python
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='variational')
```

```python
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='gibbs')
```

### Instantiate a topic model and estimate the optimal number of topics

Here we instantiate a NMF object, then generate plots with four metrics for estimating the optimal number of topics. Estimating the optimal number of topics may take a long time with a large corpus.

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

## Run a remote web server

1. Set up an instance using a service like AWS EC2 or GCP
1. If using AWS and you want to use a custom port, edit the Security Group inbound rules and create a custom TCP rule to allow inbound traffic on that port with Source defined as `0.0.0.0/0`
   1. This is the port on which someone will access the web app
1. Edit the file `/etc/nginx/sites-enabled/default` to contain the following
   1. If you changed changed the port in `config.ini`, update the `proxy_pass` port to be that value
   1. If you are using a custom port, change the `listen` port to be that value

```
server {
 listen 80;
 location / {
 include proxy_params;
 proxy_pass http://127.0.0.1:5000;
 }
}
```
