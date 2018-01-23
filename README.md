# TOEFL
It is a simple script to analyze TOEFL like passages and extract topic words.
By default, it extracts 10 topics, each contains 30 words; these values can be overridden check **usage** section.

## Libraries Installation

A simple way is to install Anaconda by running an installer, check [here](https://conda.io/docs/user-guide/install/index.html#regular-installation). Then, download required corpus by running the following script on IPython or python shell.

```
import nltk
nltk.download()
```

## Project Setup
Download and navigate to project folder using the following commands on shell.
```
git clone "https://github.com/Defrawy/toefl"
cd toefl
```

## Usage
On shell, run the following command. The script will create a subfolder `./topics` holds text files that contain topic words.

```
python toefl.py [options]
```

### Options
A set of options are available to override default values. To use it, precede it by double dashes (`--`) _like_ ```python toefl.py --n_topics=10```

Option                      | Description				
----------------------------|------------------------
  n_ftrs=N_FEATURES	| Build a vocabulary list that only consider the top max features ordered by term frequency across the corpus.
  n_tpcs=N_TOPICS   	|Number of topics to be extracted.
  n_tpwrds=N_TOP_WORDS |Number of words to represent a single topic.
  max_df=MAX_DF       	|Ignore terms that have a document frequency strictly higher than the given threshold. If float, the parameter represents a proportion of documents, integer absolute count.
  min_df=MIN_DF       	|Ignore terms that have a document frequency strictly lower than the given threshold. If float, the parameter represents a proportion of documents, integer absolute count.
  alpha=ALPHA         	|Constant that multiplies the regularization terms. Set it to zero to have no regularization.
  l1_ratio=L1_RATIO   	|Regularization parameter, with 0 <= l1_ratio <= 1.

## Acknowledgements
- The rich documentation and examples developed by sklearn team.
- Building Machine Learning Systems with Python by Luis Pedro Coelho and Willi Richert.

## License
The content of this repository is licensed under a [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)
