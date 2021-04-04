"""
Written by user: samhaug
2021 Apr 04 11:24:30 AM
"""
from gensim.parsing.porter import PorterStemmer
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from glob import glob
from os import listdir
from os.path import join
from pandas import read_csv

keep_list = ["aren't", 'couldn', "couldn't", 'didn', "didn't",
            'doesn', "doesn't", 'isn', "isn't", 'shouldn', 
            "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
            'won', "won't", 'wouldn', "wouldn't", 'not']

add_list = ['[', ']', "'re", "'s", "n't", "'ve", "'ll", 'I',
            '-', '``', "''", '__', '/', "'m", 'applause', '&',
            '@', 'weedmaps.com', 'email', '”', '“', ';', '’', '—',
            'hello']

class YoutubeCorpus():
    '''
    Provide path to directory of json files. Json files contain stripped
    youtube closed-captions.
    '''
    def __init__(self, path):
        self.path = path
        self.files = listdir(path)
        stop_list = stopwords.words('english')
        stop_list = [ w for w in stop_list if w not in keep_list ]
        for w in add_list:
            stop_list.append(w)
        self.stop_list = stop_list

    def __iter__(self):
        p = PorterStemmer()
        idx = 0
        for f in self.files:
            fname = join(self.path, f)
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
                        not in self.stop_list and w not in punctuation \
                        and not w.isnumeric()]
                desc_stem = p.stem_documents(desc_filter)
                key = base+'_'+str(i)
                yield TaggedDocument(words=desc_stem, tags=[idx, key])
                idx += 1
    
class DescriptionCorpus():
    '''
    Provide path to csv file of weedmaps decriptions
    '''
    def __init__(self, path):
        self.df = read_csv(path)
        stop_list = stopwords.words('english')
        stop_list = [ w for w in stop_list if w not in keep_list ]
        for w in add_list:
            stop_list.append(w)
        self.stop_list = stop_list

    def __iter__(self):
        p = PorterStemmer()

        for i in range(len(self.df)-1):
            key = self.df['slug'].iloc[i+1]
            desc = self.df['description'].iloc[i].lower()
            desc_token = word_tokenize(desc)
            desc_filter = [ w for w in desc_token if w not in \
                    self.stop_list and w not in punctuation and not \
                    w.isnumeric() ]
            desc_stem = p.stem_documents(desc_filter)
            
            yield TaggedDocument(words=desc_stem, tags=[i, key])






            
