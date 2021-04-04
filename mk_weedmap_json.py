"""
Written by user: samhaug
2021 Mar 29 21:57:29

"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.porter import PorterStemmer
from glob import glob
import json
import random
import pandas as pd
from string import punctuation

# Read description file
df = pd.read_csv('./data/weedmaps/cleaned_weedmaps_descriptions.csv')

p = PorterStemmer()

stop_list = stopwords.words('english')
keep_list = [ "aren't", 'couldn', "couldn't", 'didn', "didn't",
              'doesn', "doesn't", 'isn', "isn't", 'shouldn', "shouldn't",
              'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't", 'not' 
            ]

stop_list = [ w for w in stop_list if w not in keep_list ]

# Add these symbols to the stop word list
add_list = [ '[', ']', "'re", "'s", "n't", "'ve", "'ll", 'I',
             '-', '``', "''", '__', '/', "'m", 'applause', '&',
             '.', '’', ',' ,'(', ')', '--', "'", '@', 'weedmaps.com',
            'email', '”', '“', ';' '’']

for word in add_list:
    stop_list.append(word)


sentence_list = [] 
key_list = []

for i in range(len(df)-1):
    #key and descrip are offset by 1
    key = df['slug'].iloc[i+1]
    desc = df['description'].iloc[i].lower()
    key_list.append(key)
    desc = p.stem_sentence(desc)
    desc_token = word_tokenize(desc)
    desc_filter = [ w for w in desc_token if w not in stop_list 
            and w not in punctuation ] 
    sentence_list.append(desc_filter)
    
print('Running phrase model')
phrase_model = Phrases(sentence_list, min_count=10, 
            threshold=20, connector_words=ENGLISH_CONNECTOR_WORDS)
    
weedmap_dict = {}
for i in range(len(sentence_list)):
    new_phrase = phrase_model[sentence_list[i]]
    weedmap_dict[key_list[i]] = new_phrase
    
print("saving_json")
with open('clean_weedmaps.json', 'w') as fp:
    json.dump(weedmap_dict, fp)





