import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models.phrases import npmi_scorer

from sklearn.decomposition import NMF
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class Enrich:
    '''
    Collection of methods used to enrich initial dictionary (handcrafted by mix and match) using information from corpus
    
    Args:
        dictionary (dataframe): initial dictionary
        genre (array): genres of interest 

    Attributes:
        genre (array): genres of interest 
        dict1 (dictionary): initial dictionary as a dictionary instead of a dataframe
        seed_topic_list (list): list of seed topics for seeded topic modeling from initial dictionary
        vectorizer (sklearn object): CountVectorizer on pre-processed corpus
        tf_dtm (array): term frequency document-term matrix 
        vocab (array): array of feature names in document-term matrix
        new_topic_keywords(list): additional new keywords that converged around seeded topics after seeded topic modeling		
    '''
    def __init__ (self, dictionary, genre):
        self.genre = genre
        self.dict1 = {}

        for i in range(len(self.genre)):
            self.dict1[self.genre[i]] = list(set([keyword for keyword in dictionary[self.genre[i]] if not pd.isna(keyword)]))

        self.seed_topic_list = [self.dict1[topic] for topic in self.genre]      
        self.vectorizer = None
        self.tf_dtm = None
        self.vocab = None
        
        self.new_topic_keywords = None

    def get_dtm(self,corpus, min_df, max_df):
        ''' Get tfidf document-term matrix from pre-processed corpus
        
        Args:
            corpus (series): panda series containing documents in pre-processed corpus
            min_df (int): parameter in CountVectorizer for minimum documents that keywords should appear in 
            max_df (int): parameter in CountVectorizer for maximum documents that keywords should appear in 
            
        Returns:
            dtm (array): tfidf document-term matrix
        '''
        self.vectorizer = CountVectorizer(ngram_range=(1, 1), min_df =min_df, max_df = max_df)
        self.vectorizer.fit(corpus)
        self.tf_dtm = self.vectorizer.transform(corpus)

        transformer =  TfidfTransformer()
        dtm = transformer.fit_transform(self.tf_dtm)
        
        self.vocab = self.vectorizer.get_feature_names()
        
        return dtm
    
    def get_baseline_score(self, movies, cutoff = 1):
        ''' Get baseline performance metrics (precision,recall, F1) for keyword matching method
        
        Args:
            movies (dataframe): pandas dataframe of pre-processed dataset
            cutoff (int): threshold number of keywords for a movie to belong to a genre
            
        Returns:
            base_yres_test (dataframe): Baseline performance metrics for each genre
        '''
        
        indices = np.array(range(movies.shape[0]))

        # 20% hold-out for test data
        train_idx0, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        n = len(self.dict1.keys())
        topic_list = [self.dict1[self.genre[i]] for i in range(len(self.genre))]
        dict1_list = [[keyword for keyword in topic if not pd.isna(keyword)] for topic in topic_list ]
        dict1_list_idx= [[self.vocab.index(keyword) for keyword in topic if not pd.isna(keyword)] for topic in topic_list]

        doc_topic = pd.DataFrame(np.zeros((self.tf_dtm.shape[0],n)),columns = self.genre)
        for i in range(n):
            keywords_all_doc = self.tf_dtm[:,dict1_list_idx[i]].toarray()
            keywords_all_doc[keywords_all_doc > 0] = 1
            doc_topic.iloc[:,i] = keywords_all_doc.sum(axis=1)

        ypred= (doc_topic>=cutoff).astype(int)
        ypred_test = ypred.iloc[test_idx,:]
        ytrue_test = movies.iloc[test_idx,1:]

        base_yres_test = pd.DataFrame(np.zeros((3,len(self.genre))), index = ['Precision','Recall', 'F1-score'],columns = self.genre)
        for i in range(len(self.genre)):

            base_yres_test.iloc[0,i] = round(precision_score(ytrue_test.iloc[:,i], ypred_test.iloc[:,i]),4) 
            base_yres_test.iloc[1,i] = round(recall_score(ytrue_test.iloc[:,i], ypred_test.iloc[:,i]),4)
            base_yres_test.iloc[2,i] = round(f1_score(ytrue_test.iloc[:,i], ypred_test.iloc[:,i]),4)
        
        return base_yres_test
    
    def get_full_cooccurence_matrix(self):
        ''' Get full co-occurence(npmi) matrix showing pairwise npmi scores for every word in the corpus vocabulary
        
        Args:
            None
            
        Returns:
            npmi_full (dataframe): pandas dataframe for co-occurence(npmi) matrix 
        '''
        
        #word-word cooccurence matrix
        X = self.tf_dtm
        X[X > 0] = 1 
        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
        Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
        Xc = Xc.toarray()

        #npmi matrix
        word_count = self.tf_dtm.sum(axis=0).tolist()[0]
        corpus_word_count = self.tf_dtm.sum(axis=0).sum()

        pa = np.array(word_count).reshape(len(word_count),1) / corpus_word_count
        pb = np.array(word_count).reshape(1,len(word_count)) / corpus_word_count
        pab = Xc / corpus_word_count
        valid_pab = (pab != 0).astype(int)

        npmi = np.log((pab+2*(1-valid_pab))/ (pa * pb)) / -np.log(pab+2*(1-valid_pab))
        npmi = npmi*valid_pab

        npmi = abs(npmi.clip(min = 0))
        npmi_full = pd.DataFrame(npmi, index = self.vocab, columns = self.vocab)
        
        return npmi_full
        
    def generate_virtual_doc(self, npmi_full, percentile = 70):
        ''' Generate virtual document for each seeded keywords
        Documents contains keywords from corpus vocabulary that highly co-occured with seeded keywords.
        Serve as a method to restrict the npmi matrix.

        Args:
            npmi_full (array): full npmi matrix
            percentile (int): cutoff threshold for keywords to be selected for virtual documents
            
        Returns:
            vodc (list): list of virtual documents
        '''
        
        # Generation of virtual documents
        npmi_sel = npmi_full.copy(deep=True)
        seed_words = [items for substring in self.seed_topic_list for items in substring]

        npmi_sel.index = npmi_full.columns
        npmi_sel = npmi_sel.loc[seed_words,:]

        cutoff_lst = []
        for i in range(len(seed_words)):
            cutoff_lst.append(np.percentile(npmi_sel.iloc[i,:].values[np.nonzero(npmi_sel.iloc[i,:])],percentile))

        vdoc = [' '.join([seed_words[i]]+[self.vocab[idx] for idx,val in enumerate(npmi_sel.iloc[i,:]) if val>= np.max(cutoff_lst)]) for i in range(len(seed_words))]
        
        return vdoc        
        
    def get_restricted_npmi_vectors(self, vdoc, npmi_full, size = 300):
        ''' Restrict npmi matrix by virtual document and perform dimension reduction  

        Args:
            vdoc (list): list of virtual documents
            npmi_full (array): full npmi matrix
            size (int): final length for dimension reduction
            
        Returns:
            npmi_embed (array): restricted npmi matrix
            vdoc_vocab (array): keywords from virtual documents
        '''        
        # Restrict npmi matrix
        vectorizer = CountVectorizer(ngram_range=(1, 1))
        vectorizer.fit(vdoc)
        vdoc_vocab = vectorizer.get_feature_names()

        npmi_vdoc = npmi_full.copy(deep=True)

        npmi_vdoc = npmi_vdoc.loc[:,vdoc_vocab]
        npmi_vdoc.index = npmi_full.columns
        npmi_vdoc = npmi_vdoc.loc[vdoc_vocab,:]
        
        #dimension reduction using matrix factorization
        nmf = NMF(n_components = size, random_state =42,alpha=1, l1_ratio=0.0)
        nmf.fit(npmi_vdoc)

        npmi_embed = nmf.transform(npmi_vdoc)
        
        return npmi_embed, vdoc_vocab
    
    def customized_nmf(self, npmi_embed, vdoc_vocab):
        ''' Non-negative matrix factorization (with customized H matrix) on restricted npmi matrix

        Args:
            npmi_embed (array): restricted npmi matrix
            vdoc_vocab (array): keywords from virtual documents
            
        Returns:
            nmf (sklearn object): fitted nmf model
        '''        
        n = len(self.dict1.keys())
        
        nmf = NMF(n_components=n, random_state =42, init='custom',alpha=1, l1_ratio=0.0)
        dtm_used = np.transpose(npmi_embed)

        # Customize W and H matrices
        avg = np.sqrt(dtm_used.mean() /n)
        seed_idx_list = [[vdoc_vocab.index(seed) for seed in self.seed_topic_list[i]] for i in range(len(self.seed_topic_list))]
        H = np.zeros((n,dtm_used.shape[1]))
        for i in range(len(self.seed_topic_list)):
             for idx in seed_idx_list[i]:
                    H[i,idx] = avg*100

        W = avg * np.random.RandomState(42).randn(dtm_used.shape[0], n)
        W = np.abs(W)

        nmf.fit(dtm_used,H=H,W=W)
        
        return nmf
        
    def new_words(self, nmf, vdoc_vocab, n_words = 20):
        '''Get additional new words that converged around seeded keywords from nmf topic modeling

        Args:
            nmf (sklearn object): fitted nmf model
            vdoc_vocab (array): keywords from virtual documents
            n_words (int): top number of words to consider from nmf model
            
        Returns:
            new_words_df (dataframe): dataframe to check the additional new words found
        '''  
        keywords = np.array(vdoc_vocab)
        self.new_topic_keywords = []
        for topic_weights in nmf.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words+1]
            self.new_topic_keywords.append(keywords.take(top_keyword_locs))

        # Topic - New Keywords Dataframe
        new_words = [[new_word for new_word in list(self.new_topic_keywords[i]) if new_word not in self.dict1[self.genre[i]]] for i in range(len(self.genre))]
        new_words_df = pd.DataFrame(new_words,index=self.genre).T
        
        return new_words_df
    
    def pruning(self, npmi_full, vdoc_vocab, cutoff = 1):
        '''Prune keywords from enriched dictionary that highly occured with keywords in other topics
        
        Args:
            npmi_full (array): full npmi matrix
            vdoc_vocab (array): keywords from virtual documents
            cutoff (float): npmi threshold for pruning. Set cutoff = 1 if no pruning required.
            
        Returns:
            dict2_sel (dictionary): enriched dictionary as a dictionary for downstream processing
            dict2_list_idx (list): list of vocabulary index for keywords in each genre
            enriched_dict: (dataframe): enriched dictionary as a dataframe for output and checking 
        ''' 
        
        # Remove dictionary words that highly co-occured in other topics
        dict2 = {self.genre[i]:list(set(list(self.dict1[self.genre[i]]) + list(self.new_topic_keywords[i]))) for i in range(len(self.genre))}
        dict2_sel = {}

        for i in range(len(self.genre)):
            potential = dict2[self.genre[i]]
            other_words = [[word for word in dict2[self.genre[n]] if word not in potential] for n in range(len(self.genre))]
            other_words = [items for substring in other_words for items in substring]
            other_words_idx = [vdoc_vocab.index(word) for word in other_words]

            dict2_sel[self.genre[i]] = []

            for j in range(len(potential)):
                npmi_val = npmi_full.iloc[vdoc_vocab.index(potential[j]),other_words_idx]
                if np.sum(npmi_val>=cutoff) == 0: dict2_sel[self.genre[i]].append(potential[j])

        topic_list = [dict2_sel[self.genre[i]] for i in range(len(self.genre))]
        dict2_list_idx= [[self.vocab.index(keyword) for keyword in topic if not pd.isna(keyword)] for topic in topic_list]
        
        enriched_dict = pd.DataFrame([dict2_sel[self.genre[i]] for i in range(len(self.genre))],index=self.genre).T
                
        return dict2_sel, dict2_list_idx, enriched_dict



    
   