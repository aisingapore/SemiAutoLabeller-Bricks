import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB

class Label:
    '''
    Collection of methods used to automatically label unlabeled documents based on enriched dictionary
    
    Args:
        dictionary (dictionary): enriched dictionary
        genre (array): genres of interest 

    Attributes:
        genre (array): genres of interest 
        dict2_sel (dictionary): enriched dictionary 
        vectorizer (sklearn object): CountVectorizer on pre-processed corpus
        tf_dtm (array): term frequency document-term matrix 
        vocab (array): array of feature names in document-term matrix
    '''

    def __init__ (self, dictionary, genre):
        self.genre = genre
        self.dict2_sel = dictionary
        
        self.vectorizer = None
        self.tf_dtm = None
        self.vocab = None
        
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
              
    def generate_virtual_doc(self, npmi_full, percentile = 50):
        ''' Generate virtual document for each seeded keywords
        Documents contains keywords from corpus vocabulary that highly co-occured with seeded keywords.
        Serve as a method to restrict the npmi matrix.

        Args:
            npmi_full (array): full npmi matrix
            percentile (int): cutoff threshold for keywords to be selected for virtual documents
            
        Returns:
            vodc (list): list of virtual documents
        '''	    
        npmi_sel = npmi_full.copy(deep=True)
        dict2_topic_list = [self.dict2_sel[self.genre[i]] for i in range(len(self.genre))]
        dict2_list_idx= [[self.vocab.index(keyword) for keyword in topic if not pd.isna(keyword)] for topic in dict2_topic_list]
        seed_words = [substring for item in dict2_topic_list for substring in item]

        cutoff_lst = []
        for i in range(len(seed_words)):
            cutoff_lst.append(np.percentile(npmi_sel.iloc[i,:].values[np.nonzero(npmi_sel.iloc[i,:])],percentile))

        vdoc = [' '.join([seed_words[i]]+[self.vocab[idx] for idx,val in enumerate(npmi_sel.iloc[i,:]) if val>= np.max(cutoff_lst)]) for i in range(len(seed_words))]
        
        return vdoc
        
    def get_restricted_npmi_vectors(self, vdoc, npmi_full, size = 1024):
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
        nmf = NMF(n_components=size, random_state =42,alpha=1, l1_ratio=0.0)
        nmf.fit(npmi_vdoc)

        npmi_embed = nmf.transform(npmi_vdoc)
        npmi_embed = pd.DataFrame(np.transpose(npmi_embed),columns = npmi_vdoc.columns)
        
        return npmi_embed, vdoc_vocab
        
    def compute_doc_vectors(self, npmi_embed, vdoc_vocab):
        ''' Compute document vectors from npmi matrix  

        Args:
            npmi_embed (array): restricted npmi matrix
            vdoc_vocab (array): keywords from virtual documents
            
        Returns:
            doc_npmi_embed (array): document vectors concatenated as a matrix
        '''
        vdoc_idx = [self.vocab.index(word) for word in vdoc_vocab]
        doc_npmi_embed = np.dot(self.tf_dtm.toarray()[:,vdoc_idx],np.transpose(npmi_embed.values))

        # divde by total length of doc, longer length more likely to have more keywords
        doc_npmi_embed = np.array(np.divide(doc_npmi_embed,self.tf_dtm.sum(axis=1).reshape(-1,1))) 

        # docs with no topic keyword = zero embedding
        doc_npmi_embed[np.isnan(doc_npmi_embed)]=0 
        
        return doc_npmi_embed
    
    def seed_doc(self,dict2_list_idx, top = 10):
        ''' Get seeded documents that contained the most number of keywords from enriched dictionary

        Args:
            dict2_list_idx (list): list of vocabulary index for keywords in each genre
            top (int): top number of documents with most number of keywords from enriched dictionary to be used as seeded documents
            
        Returns:
            doc_seed_idx (list): index of seeded documents
        '''
        n = len(self.genre)
        
        doc_topic = pd.DataFrame(np.zeros((self.tf_dtm.shape[0],n)),columns = self.genre)
        for i in range(len(self.genre)):
            keywords_all_doc = self.tf_dtm[:,dict2_list_idx[i]].toarray()
            keywords_all_doc[keywords_all_doc > 0] = 1
            doc_topic.iloc[:,i] = keywords_all_doc.sum(axis=1)

        doc_seed_idx= []

        for i in range(len(self.genre)):
            doc_seed_idx.append(doc_topic.iloc[:,i].sort_values(ascending=False).index[:top].tolist())
            
        return doc_seed_idx
    
    def customized_nmf(self, doc_npmi_embed, doc_seed_idx):
        ''' Non-negative matrix factorization (with customized H matrix) on restricted npmi matrix

        Args:
            npmi_embed (array): restricted npmi matrix
            doc_seed_idx (list): index of seeded documents
            
        Returns:
            nmf (sklearn object): fitted nmf model
        '''      
        n = len(self.genre)        
        
        nmf = NMF(n_components=n, random_state = 42, init='custom')
        dtm_used = np.transpose(doc_npmi_embed)

        # Customize W and H matrices

        avg = np.sqrt(dtm_used.mean()/n)
        H = np.zeros((n,dtm_used.shape[1]))
        for i in range(len(doc_seed_idx)):
             for idx in doc_seed_idx[i]:
                    H[i,idx] = avg*100

        W = avg * np.random.RandomState(42).randn(dtm_used.shape[0], n)
        W = np.abs(W)

        nmf.fit(dtm_used,H=H,W=W)
        
        return nmf
        
    def auto_label_classifier(self, nmf, movies, classifier, m = 1, min_df = 3, max_df = 300):
        ''' Automatically label unlabeled documents by predicting the labels based on documents that most/least likely belong to a category
        Documents that most/least likely belong to a category were identified based on nmf topic scores

        Args:
            nmf (sklearn object): fitted nmf model
            movies (dataframe): pandas dataframe of pre-processed dataset
            classifier (sklearn classifier): classifer to be trained for prediction
            m (float): for setting threshold to identify documents that most/least likely belong to a category based on nmf topic scores
            min_df (int): parameter in CountVectorizer for minimum documents that keywords should appear in 
            max_df (int): parameter in CountVectorizer for maximum documents that keywords should appear in 
            
        Returns:
            ypred (dataframe): predicted labels for each document for each genre
        ''' 
        
        ypred = pd.DataFrame(np.zeros(np.transpose(nmf.components_).shape),columns = self.genre)

        for i in range(len(self.genre)):

            out = np.transpose(nmf.components_)[:,i]

            idx_0 = [idx for idx,val in enumerate(out) if val <= max((np.mean(out) - m*np.std(out)),0)]
            idx_1 = [idx for idx,val in enumerate(out) if val > (np.mean(out) + m*np.std(out))]
            ypred.iloc[idx_1,i] = 1

            test_idx = [x for x in list(range(ypred.shape[0])) if x not in (idx_1+idx_0)]
            #stop if seperable using label pos when topic score > 0 and label neg when topic score = 0 
            if len(test_idx) == 0 : continue

            np.random.seed(42)
            idx_label_pos = np.array(idx_1)
            idx_label_neg = np.random.choice(idx_0, len(idx_1), replace=False)
            
            # get train df('labeled' documents) and val df (remaining documents)
            df_trn = movies.iloc[np.concatenate([idx_label_pos,idx_label_neg]),0:1]
            df_trn['label'] = 0
            df_trn.iloc[df_trn.index.get_indexer(idx_label_pos),1] = 1
            df_val = movies.iloc[test_idx,0:1]
            
            #get train and val dtm 
            corpus = df_trn['overview']
            vectorizer = CountVectorizer(ngram_range=(1, 1), min_df =min_df, max_df = max_df)
            vectorizer.fit(corpus)
            tf_dtm = vectorizer.transform(corpus)
            transformer =  TfidfTransformer()
            dtm = transformer.fit_transform(tf_dtm)

            corpus_val = df_val['overview']
            tf_dtm_val = vectorizer.transform(corpus_val)
            transformer =  TfidfTransformer()
            dtm_val = transformer.fit_transform(tf_dtm_val)
            
            #model training and predict labels on remaining documents
            classifier.fit(dtm, df_trn.iloc[:,1])
            df_val['label'] = classifier.predict(dtm_val) 

            ypred.iloc[df_val.loc[df_val['label']==1,:].index.tolist(),i] = 1
            
        return ypred
