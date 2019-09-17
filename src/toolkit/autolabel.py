
import src.helper
from src.helper.preprocess import Preprocess, replace_bigrams
from src.helper.enrich import Enrich
from src.helper.topic_model import Topic_model
from src.helper.label import Label
from src.helper.supervised import Supervised

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from src.helper.supervised import Supervised

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier

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

        self.enrich = enrich

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

def check_labels(corpus, labels):
    """ removes keywords which are not contained within the corpus
    
    Arguments:
        corpus {DataFrame} -- dataframe of corpus
        labels {DataFrame} -- dataframe of the labels
    """

    # convert corpus to a set
    values = [word for sentences in corpus.values.tolist() for sentence in sentences for word in sentence.split(" ")]
    corpus_set = set(values)
    
    # remove word from labels if not exist in corpus
    labels = labels.copy(deep=True)
    for i in range(len(labels.values)):
        for j in range(len(labels.values[i])):
            word = labels.values[i][j]
            if word not in corpus_set and word is not np.nan:
                labels.values[i][j] = np.nan
                print("{} is not in the input corpus. It is removed from dictionary".format(word))

    return labels

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

class Evaluator():
    """ Class to evaluate performance of the auto labeller
    """
    def __init__(self):
        return
        
    def evaluate_predictions(self, data, ypred, labels, split=0.2, random_state=42):
        """ evaluates the prediction using precision, recall and f1 score
        
        Arguments:
            data {DataFrame} -- original input dataset
            ypred {DataFrame} -- predicted results
            labels {DataFrame} -- dictionary of labels
        
        Keyword Arguments:
            split {float} -- the train test split (default: {0.2})
            random_state {int} -- numeric seed for random state (default: {42})
        
        Returns:
            DataFrame -- precision, recall and f1 score for the model
        """
        indices = np.array(range(data.shape[0]))
        topics = labels.columns

        # 20% hold-out for test data
        train_idx, test_idx = train_test_split(
            indices, test_size=split, random_state=random_state)

        self.train_idx, self.test_idx = train_idx, test_idx

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

    def compare_to_other_models(self, score, data, labels):
        """ Code to compare performance with other models
        
        Arguments:
            score {DataFrame} -- results of auto labelling tool
            data {DataFrame} -- original data input
            labels {DataFrame} -- labelling dictionary
        
        Returns:
            {DataFrame} -- precision, recall and f1 scores of various models
        """
            
        # Supervised Learning document-term matrix using 10% train data for supervised learning
        supervised = Supervised(self.test_idx, labels.columns)
        supervised.get_dtm(data, self.train_idx, min_df=3, max_df=300, text_column='content')

        # Multilayer perceptron classifier
        mlp = MLPClassifier(hidden_layer_sizes=(512,64,), random_state=42)
        mlp_yres_test = supervised.classifier(mlp)

        # Gradient boosted trees
        gbm = GradientBoostingClassifier(random_state=42)
        gbm_yres_test = supervised.classifier(gbm)

        # Random Forest
        rf = RandomForestClassifier(n_estimators = 500, random_state=42, n_jobs=-1)
        rf_yres_test = supervised.classifier(rf)

        # combine results into a summary table
        summary = pd.concat([np.round(score.mean(axis=1),3),np.round(mlp_yres_test.mean(axis=1),3),np.round(gbm_yres_test.mean(axis=1),3),np.round(rf_yres_test.mean(axis=1),3)],axis=1)
        summary.columns = ['Automatic Labeling','MLP Neural Network', 'Gradient Boosted Trees', 'Random Forest']

        return summary