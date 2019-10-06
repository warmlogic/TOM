from setuptools import setup, find_packages
import tom_lib

version = tom_lib.__version__

setup(
    name='tom_lib',
    version=version,
    packages=find_packages(),
    author="Adrien Guille, Pavel Soriano, Matt Mollison",
    author_email="matt.mollison@gmail.com",
    description="A library for topic modeling and browsing (forked by MVM)",
    long_description=open('README.rst').read(),
    url='http://mediamining.univ-lyon2.fr/people/guille/tom.php',
    download_url=f'http://pypi.python.org/packages/source/t/tom_lib/tom_lib-{version}.tar.gz',
    license="MIT",
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing'
    ],
    install_requires=[
        'dash',
        'dash-daq',
        'dash-table',
        'gensim',
        'lda',
        'matplotlib',
        'networkx',
        'nltk',
        'numpy',
        'openpyxl',
        'pandas',
        'plotly',
        'python-dotenv',
        'scikit-learn',
        'scipy',
        'seaborn',
        'smart_open',
    ]
)
