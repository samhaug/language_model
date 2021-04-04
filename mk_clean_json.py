"""
Written by user: samhaug
2021 Apr 04 01:55:06 PM
"""

from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from glob import glob
from os import listdir
from os.path import join
from pandas import read_csv
import json

# Each json file will have at most json_size documents
json_size = 1000

stop_list = stopwords.words('english')

keep_list = [ "aren't", 'couldn', "couldn't", 'didn', "didn't",
            'doesn', "doesn't", 'isn', "isn't", 'shouldn', 
            "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
            'won', "won't", 'wouldn', "wouldn't", 'not' ]

add_list = [ '[', ']', "'re", "'s", "n't", "'ve", "'ll", 'I',
            '-', '``', "''", '__', '/', "'m", 'applause', '&',
            '@', 'weedmaps.com', 'email', '”', '“', ';', '’', '—',
            'hello' ]

stop_list = [ w for w in stop_list if w not in keep_list ]

for word in add_list:
    stop_list.append(word)

def mk_clean_youtube(path, clean_path):
    '''
    path: path to directories containing raw json files stripped 
          from youtube CCs
    clean_path: directory where clean json files will be written
    '''
    p = PorterStemmer()
    idx = 0
    j = 1
    files = listdir(path)
    youtube_dict = {}
    for f in files:

        fname = join(path, f)
        try:
            lines = open(fname).readlines()[0]
        except IndexError:
            continue
        prod_list = lines.strip().split(':')[1:]
        base = fname.split('/')[-1].split('.')[0].strip('_')
        
        for i in range(len(prod_list)):
            desc = prod_list[i].split(',')[0]
            desc = desc.replace('[Music]','')
            desc_token = word_tokenize(desc.lower())
            desc_filter = [ w for w in desc_token if w \
                    not in stop_list and w not in punctuation \
                    and not w.isnumeric() ]
            desc_stem = p.stem_documents(desc_filter)
            key = base+'_'+str(i)
            # only save document if it has at least 50 tokenized words
            if len(desc_stem) <= 50:
                continue
            youtube_dict[key] = desc_stem

            # write out json file if dictionary reaches 1000 entries
            if idx == json_size:
                print("Saving youtube_{}.json".format(str(j)))
                with open(join(clean_path,'youtube_{}.json'.format(
                        str(j))), 'w') as fp:
                    json.dump(youtube_dict, fp)
                youtube_dict = {}
                j+=1
                idx=0
            idx+=1 

    with open(join(clean_path,'youtube_{}.json'.format(
            str(j))), 'w') as fp:
        json.dump(youtube_dict, fp)
            

def mk_clean_weedmaps(path, clean_path):
    '''
    path: path to csv file with cleaned weedmaps reviews
    clean_path: directory where clean json files will be written
    '''
    df = read_csv(path)
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


    



















