"""
Written by user: samhaug
2021 Mar 30 15:50:55
"""
import json
import gensim
from os.path import isfile, join
import collections

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

doc_dict = json.load(open('clean_weedmaps.json','r'))
youtube_dict = json.load(open('../scrapped_youtube_captions/clean_youtube.json','r'))
doc_dict.update(youtube_dict)


documents = [ TaggedDocument(doc_dict[key], [i]) 
        for i, key in enumerate(doc_dict) ]

model = gensim.models.doc2vec.Doc2Vec(
        vector_size=128, min_count=50, epochs=10)

model.build_vocab(documents)

print("training")
model.train(documents, total_examples=model.corpus_count, 
        epochs=model.epochs)

print("Evaluating")
ranks = []
second_ranks = []
for doc_id in range(len(documents)):
    inferred_vector = model.infer_vector(documents[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])


counter = collections.Counter(ranks)
print(counter)


print('Document ({}): «{}»\n'.format(doc_id, ' '.join(
        documents[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
#for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2),  ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))

print('')
print('#'*150)
print('#'*150)
print('')

print('Document ({}): «{}»\n'.format(10, ' '.join(
        documents[10].words)))
inferred_vector = model.infer_vector(documents[10].words)
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2),  ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))


