from sklearn.feature_extraction.text import TfidfVectorizer

import src.helper
from src.helper.preprocess import Preprocess, replace_bigrams
from src.helper.enrich import Enrich
from src.helper.topic_model import Topic_model
from src.helper.label import Label
from src.helper.supervised import Supervised


def corpus_replace_bigrams(min_df, max_df, corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=max_df)
    vectorizer.fit(corpus)
    bigrams = [word for word in vectorizer.get_feature_names() if len(word.split(' '))==2]
    corpus_replaced = [replace_bigrams(text,bigrams) for text in corpus]
    return corpus_replaced


def preprocess(text, stopwords_path):
    preprocesser = Preprocess(text)

    preprocesser.expand_contractions()        
    preprocesser.filter_out_PERSON_named_entity()
    preprocesser.lemmatize_text()           
    preprocesser.replace_negation()

    preprocesser.remove_special_characters()
    preprocesser.remove_digits()
    preprocesser.remove_stopwords(stopwords_path)

    preprocesser.keep_pos(keep_list =['n','v','a'])
    
    return preprocesser.return_text()


def corpus_preprocess(corpus, stopwords_path):
    preprocessed_text = [preprocess(text, stopwords_path) for text in corpus]
    return preprocessed_text


class AutoLabeller():
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
        npmi_embed, vdoc_vocab = enrich.get_restricted_npmi_vectors(vdoc, npmi_full, size=300)

        # NMF on restricted npmi matrix using customize W and H matrices
        nmf = enrich.customized_nmf(npmi_embed, vdoc_vocab)

        # Potential new words to be added to dictionary
        new_words_df = enrich.new_words(nmf, vdoc_vocab, n_words=n_words)

        # Remove dictionary words that highly co-occured in other topics
        # set cutoff as 1 if pruning not required 
        dict2_sel, dict2_list_idx, enriched_labels = enrich.pruning(npmi_full, vdoc_vocab, cutoff=1)
        
        self._npmi_full = npmi_full
        self._dict2_sel = dict2_sel
        self._dict2_list_idx = dict2_list_idx
        
        return enriched_labels

    def apply(self, model, top=10):

        # Initialize label class
        label = Label(self._dict2_sel, self._genre)

        # Get document-term matrix for corpus
        dtm = label.get_dtm(self._corpus, min_df=3, max_df=300)

        # Generation of virtual doc
        vdoc = label.generate_virtual_doc(self._npmi_full, percentile=50)

        # New restricted npmi matrix
        npmi_embed, vdoc_vocab = label.get_restricted_npmi_vectors(vdoc, self._npmi_full, size=1024)

        # Compute doc embedding from npmi word embedding (simple averaged word embedding)
        doc_npmi_embed = label.compute_doc_vectors(npmi_embed, vdoc_vocab)

        # Seeded document for each topic (used as seeds to initialize custom matrices in NMF)
        doc_seed_idx = label.seed_doc(self._dict2_list_idx, top=top)

        # NMF topic modeling based on seeded documents
        nmf = label.customized_nmf(doc_npmi_embed, doc_seed_idx)

        # "label" documents with very high and very low topic score and classify remaining unlabeled documents (higher uncertainty) using a naive bayes model
        ypred = label.auto_label_classifier(nmf, self._data, model, m=0.5, min_df=3, max_df=300)

        return ypred
