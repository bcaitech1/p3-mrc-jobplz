import os
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
import json
import pandas as pd
from tqdm import tqdm
import kss
import re
# 서버 실행
es_server = Popen(['/opt/ml/elasticsearch-7.9.2/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda:os.setuid(1))

es = Elasticsearch('localhost:9200')

print(es.info())

es.indices.create(index = 'document_original_set',
                  body = {
                      'settings':{
                          'analysis':{
                              'analyzer':{
                                  'my_analyzer':{
                                      "type": "custom",
                                      'tokenizer':'nori_tokenizer',
                                      'decompound_mode':'mixed',
                                      'stopwords':'_korean_',
                                      "filter": ["lowercase",
                                                 "my_shingle_f",
                                                 "nori_readingform",
                                                 "nori_number"]
                                  }
                              },
                              'filter':{
                                  'my_shingle_f':{
                                      "type": "shingle"
                                  }
                              }
                          },
                          'similarity':{
                              'my_similarity':{
                                  'type':'BM25',
                              }
                          }
                      },
                      'mappings':{
                          'properties':{
                              'title':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              },
                              'text':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              }
                          }
                      }
                  }
                  )

print(es.indices.get('document_original_set'))


# trash_ids = [973, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 5527,
#              9079, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087, 9088, 28989, 29028,
#               31111, 37157]


with open('/opt/ml/input/data/data/wikipedia_documents.json', 'r') as f :
    # wiki_data = pd.DataFrame(json.load(f)).transpose()
    wiki = json.load(f)

new_wiki = [(vv[:10], vv.strip()) for vv in list(dict.fromkeys([v['text'] for v in wiki.values()]))]
new_wiki_title = []
new_wiki_text = []

# for current_title, current_text in tqdm(new_wiki) :
#     try :
#         te = kss.split_chunks(current_text, max_length= 1280, overlap = True)
#         for i in range(len(te)) :
#             new_wiki_title.append(current_title + str(i))
#             new_wiki_text.append(te[i].text)
#     except :
#         new_wiki_title.append(current_title)
#         new_wiki_text.append(current_text)
#         continue

print('start elastic search indexing')
for num in tqdm(range(len(new_wiki))) :
    es.index(index = 'document_original_set', body = {"title" : new_wiki[num][0], 'text' : new_wiki[num][1]})

question = '대한민국의 대통령은 누구인가?'

query = {
    'query':{
        'bool':{
            'must':[
                      {'match':{'text':question}}
            ],
            'should':[
                      {'match':{'text':question}}
            ]
        }
    }
}

doc = es.search(index='document_original_set',body=query,size=10)['hits']['hits']

print(doc)