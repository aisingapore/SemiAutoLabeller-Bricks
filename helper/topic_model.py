import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import NMF
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import gensim

class Topic_model:
    def __init__ (self):
        self.vectorizer = None
        self.tf_dtm = None

    def get_dtm(self,corpus, min_df, max_df):
        self.vectorizer = CountVectorizer(ngram_range=(1, 1), min_df =min_df, max_df = max_df)
        self.vectorizer.fit(corpus)
        self.tf_dtm = self.vectorizer.transform(corpus)

        transformer =  TfidfTransformer()
        dtm = transformer.fit_transform(self.tf_dtm)
        
        return dtm
    
    def get_nmf_coherence_score(self, dtm, topic_num):
        corpus_gensim = gensim.matutils.Sparse2Corpus(dtm, documents_columns=False)
        id2word = corpora.Dictionary.from_corpus(corpus_gensim,id2word=dict((id, word) for word, id in self.vectorizer.vocabulary_.items()))

        score_list =[]
        for n in topic_num:    
            nmf = NMF(n_components=n, random_state =42)
            nmf.fit(dtm)

            keywords = np.array(self.vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in nmf.components_:
                top_keyword_locs = (-topic_weights).argsort()[:20]
                topic_keywords.append(keywords.take(top_keyword_locs))

            text = [[item for sublist in [[keywords[i]]*v for i,v in enumerate(text) if v!=0] for item in sublist] for text in self.tf_dtm.toarray()]

            coherence_model = CoherenceModel(topics = topic_keywords, texts = text, dictionary=id2word, coherence='c_v', processes = 1)
            score_list.append(coherence_model.get_coherence())
            
        return score_list
    
    def show_topics(self, dtm, best_n, n_words):
        nmf_best = NMF(n_components = best_n,random_state =42)
        nmf_best.fit(dtm)

        #Show top n keywords for each topic
        def show_topics(vectorizer=None, model=None, n_words=20):
            keywords = np.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]
                topic_keywords.append(keywords.take(top_keyword_locs))
            return topic_keywords

        topic_keywords = show_topics(vectorizer=self.vectorizer, model=nmf_best, n_words=n_words)        

        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.to_csv('prelim_keywords.csv',index=False, header=True)
        
        return df_topic_keywords
            
            
    
        
    

