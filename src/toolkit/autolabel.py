
import src.helper
from src.helper.preprocess import Preprocess, replace_bigrams
from src.helper.enrich import Enrich
from src.helper.topic_model import Topic_model
from src.helper.label import Label
from src.helper.supervised import Supervised

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

class Preprocessor():
    """ Preprocessor holds methods to preprocess text data for auto labeller
    """

    def __init__(self):
        return

    def corpus_replace_bigrams(self, corpus, min_df, max_df):
        """ replace words within corpus that frequently occur together with a singular representation
        e.g. world war -> world_war

        Arguments:
            min_df {DataFrame} -- ???
            max_df {DataFrame} -- ???
            corpus {Series} -- Text data to be clustered

        Returns:
            List -- corpus that has its frequent bigrams replaced with a singular representation
        """
        vectorizer = TfidfVectorizer(ngram_range=(
            1, 2), min_df=min_df, max_df=max_df)
        vectorizer.fit(corpus)
        bigrams = [word for word in vectorizer.get_feature_names()
                   if len(word.split(' ')) == 2]
        corpus_replaced = [replace_bigrams(text, bigrams) for text in corpus]
        return corpus_replaced

    def _preprocess(self, text, stopwords_path):
        """ function to preprocess text by performing transformations on the text

        Arguments:
            text {String} -- a singular string of text for the preprocessor to work on
            stopwords_path {String} -- path to a dictionary of stopwords

        Returns:
            ??? -- preprocessed string
        """
        preprocesser = Preprocess(text)

        preprocesser.expand_contractions()
        preprocesser.filter_out_PERSON_named_entity()
        preprocesser.lemmatize_text()
        preprocesser.replace_negation()

        preprocesser.remove_special_characters()
        preprocesser.remove_digits()
        preprocesser.remove_stopwords(stopwords_path)

        preprocesser.keep_pos(keep_list=['n', 'v', 'a'])

        return preprocesser.return_text()

    def corpus_preprocess(self, corpus, stopwords_path):
        """ Preprocesses the entire corpus dataset

        Arguments:
            corpus {List} -- A list of text to be labelled
            stopwords_path {String} -- path to a dictionary of stopwords

        Returns:
            List -- A list of preprocessed text to be labelled
        """
        preprocessed_text = [self._preprocess(
            text, stopwords_path) for text in corpus]
        return preprocessed_text


class AutoLabeller():
    """ Contains methods to train and get labels for dataset
    """

    def __init__(self, labels, corpus, data):
        self._labels = labels
        self._corpus = corpus
        self._genre = labels.columns
        self._data = data
        return

    def train(self, n_words=20):

        enrich = Enrich(self._labels, self._genre)

        # Get document-term matrix for corpus
        dtm = enrich.get_dtm(self._corpus, min_df=3, max_df=300)

        # Get full word co-occurence matrix based on gensim npmi scorer
        npmi_full = enrich.get_full_cooccurence_matrix()

        # Generation of virtual documents
        vdoc = enrich.generate_virtual_doc(npmi_full, percentile=70)

        # Get npmi vectors (dimension=300) for each word on the restricted npmi matrix
        npmi_embed, vdoc_vocab = enrich.get_restricted_npmi_vectors(
            vdoc, npmi_full, size=300)

        # NMF on restricted npmi matrix using customize W and H matrices
        nmf = enrich.customized_nmf(npmi_embed, vdoc_vocab)

        # Potential new words to be added to dictionary
        new_words_df = enrich.new_words(nmf, vdoc_vocab, n_words=n_words)

        # Remove dictionary words that highly co-occured in other topics
        # set cutoff as 1 if pruning not required
        dict2_sel, dict2_list_idx, enriched_labels = enrich.pruning(
            npmi_full, vdoc_vocab, cutoff=1)

        self._npmi_full = npmi_full
        self._dict2_sel = dict2_sel
        self._dict2_list_idx = dict2_list_idx

        return enriched_labels

    def apply(self, model, col_name, top=10):

        # Initialize label class
        label = Label(self._dict2_sel, self._genre)

        # Get document-term matrix for corpus
        dtm = label.get_dtm(self._corpus, min_df=3, max_df=300)

        # Generation of virtual doc
        vdoc = label.generate_virtual_doc(self._npmi_full, percentile=50)

        # New restricted npmi matrix
        npmi_embed, vdoc_vocab = label.get_restricted_npmi_vectors(
            vdoc, self._npmi_full, size=1024)

        # Compute doc embedding from npmi word embedding (simple averaged word embedding)
        doc_npmi_embed = label.compute_doc_vectors(npmi_embed, vdoc_vocab)

        # Seeded document for each topic (used as seeds to initialize custom matrices in NMF)
        doc_seed_idx = label.seed_doc(self._dict2_list_idx, top=top)

        # NMF topic modeling based on seeded documents
        nmf = label.customized_nmf(doc_npmi_embed, doc_seed_idx)

        # "label" documents with very high and very low topic score and classify remaining unlabeled documents (higher uncertainty) using a naive bayes model
        ypred = label.auto_label_classifier(
            nmf, self._data, col_name, model, m=0.5, min_df=3, max_df=300)

        return ypred


def recommend_words(corpus, topic_num=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], min_df=3, max_df=300):
    """ recommend words is the method used in the first round of recommendation for a list of words to
    consider to be labelled
    topic_num is the range of numbers to test

    Returns:
        tuple -- contains the topic_model, document term matrix and best number of topics to cluster
    """
    topic_model = Topic_model()

    dtm = topic_model.get_dtm(corpus, min_df, max_df)

    # get list of coherence scores of topic numbers
    score_list = topic_model.get_nmf_coherence_score(dtm, topic_num)

    # get the best number of topics
    best_n = topic_num[score_list.index(max(score_list))]

    return topic_model, dtm, best_n


def evaluate_predictions(data, ypred, labels, split=0.2, random_state=42):
    indices = np.array(range(data.shape[0]))
    topics = labels.columns

    # 20% hold-out for test data
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)

    ypred_test = ypred.iloc[test_idx, :]
    ytrue = data.iloc[:, 1:]
    ytrue_test = ytrue.iloc[test_idx, :]

    auto_yres_test = pd.DataFrame(np.zeros((3, len(topics))),
                                  index=['Precision', 'Recall', 'F1-score'],
                                  columns=topics)

    for i in range(len(topics)):
        precision = precision_score(ytrue_test.iloc[:, i],
                                    ypred_test.iloc[:, i])
        auto_yres_test.iloc[0, i] = round(precision, 4)

        recall = recall_score(ytrue_test.iloc[:, i], ypred_test.iloc[:, i])
        auto_yres_test.iloc[1, i] = round(recall, 4)

        f1 = f1_score(ytrue_test.iloc[:, i], ypred_test.iloc[:, i])
        auto_yres_test.iloc[2, i] = round(f1, 4)

    return auto_yres_test
