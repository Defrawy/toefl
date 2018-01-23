#!/usr/bin/env python

from __future__ import print_function
from time import time
from optparse import OptionParser

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import nltk
import os
import sys


# default values
N_FTRS = 5000
N_TPCS = 10
N_TP_WRDS = 30
MAX_DF = .6
MIN_DF = 1
ALPHA = .1
L1_RATIO = .5

# It is derived from TfidVectorizer class. It does not remove stop words to ensure optimal correctness of word's tag
# These words with associated tags are used to extract lemmatized words
# Stop words like "is" and "am" will be convert to "be" where it can be later removed by max_df or min_df
# As a result, other words will be categorized syntactically correct under one word.
class lemmatizedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
      analyzer = super(TfidfVectorizer, self).build_analyzer()
      return lambda doc: [WordNetLemmatizer().lemmatize(token[0], get_wordnet_pos(token[1])) for token in nltk.pos_tag(analyzer(doc))]

# configure command line options.
def init_op():
  op = OptionParser()

  op.add_option("--n_ftrs",
    action="store", type="int", dest="n_ftrs", default=N_FTRS,
    help="Build a vocabulary list that only consider the top max features ordered by term frequency across the corpus.")

  op.add_option("--n_tpcs",
    action="store", type="int", dest="n_tpcs", default=N_TPCS,
    help="Number of topics to be extracted.")

  op.add_option("--n_tpwrds",
    action="store", type="int", dest="n_tp_wrds", default=N_TP_WRDS,
    help="Number of words to represent a single topic.")

  op.add_option("--max_df",
    dest="max_df", default=MAX_DF,
    help="Ignore terms that have a document frequency strictly higher than the given threshold. If float, the parameter represents a proportion of documents, integer absolute count.")

  op.add_option("--min_df",
    dest="min_df", default=MIN_DF,
    help="Ignore terms that have a document frequency strictly lower than the given threshold. If float, the parameter represents a proportion of documents, integer absolute count.")

  op.add_option("--alpha",
    type="float", dest="alpha", default=ALPHA,
    help="Constant that multiplies the regularization terms. Set it to zero to have no regularization.")

  op.add_option("--l1_ratio",
    type="float", dest="l1_ratio", default=L1_RATIO,
    help="The regularization mixing parameter, with 0 <= l1_ratio <= 1.")
  return op

# Convert nltk post to wordnet post compliant.
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Collect topic words and save them.
def print_topic_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topic_content = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print("Topic #%d: %s..." % (topic_idx, topic_content[:60]))
        write_topic("topics/Topic #%d.txt" % topic_idx, topic_content)
    print("\nTHE COMPLETE WORD LISTS ARE SAVED ON DISK.\n")


# write content to disk.
def write_topic(name, content):
    f = open(name, "w")
    f.write(content)
    f.close()

def main():
  op = init_op()

  print()
  op.print_help()
  print()

  argv = sys.argv[1:]
  (options, args) = op.parse_args(argv)


  print("Loading passages...")
  t0 = time()
  # loading toefl reading passages
  dataset = load_files('./passages', load_content=True)
  print("Loading is done in %0.3fs." % (time() - t0))

  print("Extracting tf-idf features for NMF...")
  t0 = time()
  # Term frequency-inverse document frequency matrix
  tfidf_vectorizer = lemmatizedTfidfVectorizer(max_df=options.max_df, min_df=options.min_df,
                                     max_features=options.n_ftrs,
                                     analyzer='word')
  print("Vectorization is done in %0.3fs." % (time() - t0))
  t0 = time()
  # Term-document matrix
  tfidf = tfidf_vectorizer.fit_transform(dataset.data)
  print("Fitting is done in %0.3fs." % (time() - t0))


  print("Fitting the NMF model with tf-idf features\n")
  # Non-negative matrix factorization model
  nmf = NMF(n_components=options.n_tpcs,
            alpha=options.alpha, l1_ratio=options.l1_ratio).fit(tfidf)

  tfidf_feature_names = tfidf_vectorizer.get_feature_names()
  print_topic_words(nmf, tfidf_feature_names, options.n_tp_wrds)

if __name__ == "__main__":
  main()
