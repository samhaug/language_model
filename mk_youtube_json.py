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

p = PorterStemmer()

stop_list = stopwords.words('english')
keep_list = ["aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'isn', "isn't", 'shouldn', "shouldn't",
             'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
             'wouldn', "wouldn't", 'not']

for word in keep_list:
    stop_list.remove(word)

# Add these symbols to the stop word list
add_list = ['[', ']', "'re", "'s", "n't", "'ve", "'ll", 'I',
           '-', '``', "''", '__', '/', "'m", 'applause', '&']

for word in add_list:
    stop_list.append(word)


file_list = glob('./products/*json')
youtube_dict = {}
for filename in file_list[::20]:                              

    lines = open(filename).readlines()[0]
    prod_list = lines.strip().split(':')[1:]
    base = filename.split('/')[-1].split('.')[0].strip('_')
    print("processing ", base)
    
    sentences = [] 
    for i in range(len(prod_list)):
        desc = prod_list[i].split(',')[0]
        desc = desc.replace('[Music]','')
        desc = p.stem(desc)
        desc_token = word_tokenize(desc)
        desc_filter = [ w for w in desc_token if w not in stop_list ] 
        key = base+'_'+str(i)
        sentences.append(desc_filter)
    
    print('Running phrase model')
    phrase_model = Phrases(sentences, min_count=20, 
            threshold=20, connector_words=ENGLISH_CONNECTOR_WORDS)
    
    for i in range(len(sentences)):
        key = base+'_'+str(i)
        new_phrase = phrase_model[sentences[i]]
        youtube_dict[key] = new_phrase
    
print("saving_json")
with open('clean_youtube.json', 'w') as fp:
    json.dump(youtube_dict, fp)





