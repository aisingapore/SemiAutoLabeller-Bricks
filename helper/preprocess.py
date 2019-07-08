import pandas as pd
import numpy as np
import re
import string
import csv

import nltk
from nltk.corpus import wordnet as wn
wnl = nltk.stem.WordNetLemmatizer()

from contractions import CONTRACTION_MAP

class Preprocess:
	'''
    Collection of methods used for text preprocessing of a document
    
    Args:
        text (string): document of interest

    Attributes:
        text (string): document of interest
    '''		

    def __init__ (self, text):
        self.text = text

    def tokenize_text(self):
        '''Tokenize Text

        Args: 
			None
			
        Returns:
            tokens (list): list of tokens for document of interest
        '''
        
        tokens = nltk.word_tokenize(self.text) 
        tokens = [token.strip() for token in tokens] 
        return tokens
    
    def return_text(self):
        '''Return text as string

        Args: 
			None
			
        Returns:
            text (string): processed document of interest
        '''
        
        return self.text
    
    def expand_match(self, contraction):
        '''Helper function for expand_contractions
		
		Args: 
			contraction(dictionary): contraction mapping
			
        Returns:
            expanded_contraction (string): expanded contraction e.g. it's -> it is
        '''
        
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match) if CONTRACTION_MAP.get(match) else CONTRACTION_MAP.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]                                         
            
        return expanded_contraction

    def expand_contractions(self):
        '''Expand contractions e.g. it's -> it is from contraction mapping in contraction.py
 		Args: 
			None
			
        Returns:
            None
        '''
        
        contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), flags=re.IGNORECASE|re.DOTALL)            
        expanded_text = contractions_pattern.sub(self.expand_match, self.text)  
        expanded_text = re.sub("'", "", expanded_text)
        
        self.text = expanded_text

    def remove_special_characters(self):
        '''Remove special characters such as punctuations
 		Args: 
			None
			
        Returns:
            None
        '''
        
        tokens = self.tokenize_text()
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
        filtered_tokens = [pattern.sub('', token) for token in tokens]
        filtered_text = ' '.join([token for token in filtered_tokens if token !=''])
        
        self.text = filtered_text

    def remove_stopwords(self):
        '''Remove stopswords using nltk stopwords list and custom stopwords list in stopwords.csv file
 		Args: 
			None
			
        Returns:
            None
        '''
        
        stopword_list = nltk.corpus.stopwords.words('english') 
        stopword_list = stopword_list + pd.read_csv('stopwords.csv')['stopwords'].tolist()
        
        tokens = self.tokenize_text()
        filtered_tokens = [token for token in tokens if token not in stopword_list] 
        filtered_text = ' '.join(filtered_tokens) 
        
        self.text = filtered_text

    def remove_digits(self):
        '''Remove digits
 		Args: 
			None
			
        Returns:
            None
        '''
        
        tokens = self.tokenize_text()
        filtered_tokens = [re.sub('\d+','', token) for token in tokens]
        filtered_text = ' '.join([token for token in filtered_tokens if token !='']) 
        
        self.text = filtered_text    
    
    def filter_out_PERSON_named_entity(self):
        '''Filters out person's name from text
 		Args: 
			None
			
        Returns:
            None
        '''        
        
        tokens = self.tokenize_text()
        filtered_tokens = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(tokens)):
            if type(chunk) == tuple: filtered_tokens.append(chunk[0])
            elif chunk.label() != 'PERSON': filtered_tokens.append(chunk[0][0])
        filtered_text = ' '.join(filtered_tokens)
        
        self.text = filtered_text

    def wn_tags(self, token_pos):
        '''Helper function for pos_tag_text
 		Args: 
			token_pos (string): part-of-speech tag 
			
        Returns:
            wordnet pos tag (string): part-of-speech tag for wordnet
        '''
        
        if token_pos == 'ADJ': return wn.ADJ
        if token_pos == 'VERB': return wn.VERB
        if token_pos == 'NOUN': return wn.NOUN
        if token_pos == 'ADV': return wn.ADV
        else: return None

    def pos_tag_text(self):
        '''For part-of-speech tagging
 		Args: 
			token_pos (string): part-of-speech tag 
			
        Returns:
            tagged_text (tuple): tuple with (token, wordnet pos tag)
        '''
        
        tokens = self.tokenize_text()
        tagged_text = nltk.pos_tag(tokens,tagset='universal')
        tagged_text = [(token[0].lower(),self.wn_tags(token[1])) for token in tagged_text]
        
        return tagged_text

    def lemmatize_text(self):
        '''Lemmatize text using nltk wordnet lemmatizer. More accurate lemmatization with pos-tags.
 		Args: 
			None
			
        Returns:
            None
        '''

        pos_tagged_text = self.pos_tag_text()
        lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
        lemmatized_text = ' '.join(lemmatized_tokens) 
        
        self.text = lemmatized_text
    
    def keep_pos(self,keep_list =['n','v','a','r']):
        '''Only keep words with specified pos-tags especially for topic modeling
 		Args: 
			keeplist (list): list of pos_tag to keep
			
        Returns:
            None
        '''
        keep_tokens = [word for word,pos in self.pos_tag_text() if (pos in keep_list or pos is None)]
        keep_text = ' '.join([token for token in keep_tokens if token !=''])
        
        self.text = keep_text   
        
    def replace(self, word, pos=None):
        '''Helper function for replace_negations
 		Args: 
			word (string): negation word
			pos (string): pos tag
			
        Returns:
            antonyms (string): antonym of negation word 		
        '''
        antonyms = set()
        for syn in wn.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
        if len(antonyms) == 1: return antonyms.pop()
        else: return None

    def replace_negations(self, tokens):
        '''Helper function for replace negation
 		Args: 
			tokens (string): word tokens
			
        Returns:
            words (string): replace negation with antonym e.g. 'not happy' -> 'sad'		
        '''
        i, l = 0, len(tokens)
        words = []
        pos_tag = [pos for word,pos in self.pos_tag_text()]
        while i < l:
            word = tokens[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(tokens[i+1],pos=pos_tag[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words

    def replace_negation(self):
        '''Replace negation e.g. not happy -> sad
 		Args: 
			None
			
        Returns:
            None
        '''
                
        tokens = self.tokenize_text()
        replaced_tokens = self.replace_negations(tokens)
        replaced_text = ' '.join([token for token in replaced_tokens if token !=''])
        
        self.text = replaced_text
        
def replace_bigrams(text,bigrams):
        '''Combine frequently occured bigrams into one word using an '_' eg. 'opening hours' -> 'opening_hours'
 		Args: 
			text (string): document of interest
			bigrams (list): list of frequently occured bigrams identified by tfidf vectorizer
			
        Returns:
            replaced_text (string): document with frequently occured bigrams combined into one word
        '''
    
    def tokenize_text(text): 
        tokens = nltk.word_tokenize(text) 
        tokens = [token.strip() for token in tokens] #remove whitespace in tokens
        return tokens
    
    split_bigram = [bigram.split(' ') for bigram in bigrams]
    tokens = tokenize_text(text)
    for i in range(len(tokens)-1):
        for j in range(len(split_bigram)):
            if tokens[i] == split_bigram[j][0] and tokens[i+1] == split_bigram[j][1]:
                tokens[i] = tokens[i]+'_'+tokens[i+1]
                tokens[i+1] = ''

    replaced_text = ' '.join([token for token in tokens if token !=''])
    return replaced_text