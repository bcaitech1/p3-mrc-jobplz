import os
from subprocess import Popen, PIPE, STDOUT
from time import sleep

from elasticsearch import Elasticsearch
from numpy.core.numeric import indices

import json
import pandas as pd

from tqdm.notebook import tqdm
# from tqdm import tqdm

import re
from tqdm.auto import tqdm
import pandas as pd
import json
import os
import numpy as np
# from utils_qa import BM25

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager
import kss
import pickle

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class ElasticRetrieval:
    def __init__(self, tokenize_fn, data_path="/opt/ml/code/dongjae/mycode/data/", context_path="wikipedia_documents.json"):
        # self.query_or_dataset=query_or_dataset
        # self.topk=topk
        self.data_path=data_path
        self.data_path=context_path


    def connect(self):
        print('엘라스틱 접속')
        es_server = Popen(['/opt/ml/elasticsearch-7.12.1/bin/elasticsearch'],
                        stdout=PIPE, stderr=STDOUT,
                        preexec_fn=lambda: os.setuid(1)
                        )
        sleep(30)
        es = Elasticsearch('localhost:9200', timeout=30, max_retries=10, retry_on_timeout=True)

        print("Elastic Search 접속 확인")
        return es, es_server

    

    def retrieve(self,query_or_dataset="",topk=100):
        es,es_server = self.connect()
        
        print(es.info())

        try:
            self.indices_create(es)
        except:
            es.indices.delete('document')
            self.indices_create(es)

        with open('/opt/ml/code/data/wikipedia_documents.json', 'r') as f:
            wiki_data = pd.DataFrame(json.load(f)).transpose()

        trash_ids = [973, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 5527,
             9079, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087, 9088, 28989, 29028,
              31111, 37157]

        # for num in tqdm(range(len(wiki_data))):
        #     if num not in trash_ids:
        #         es.index(index='document', body = {"title" : wiki_data['title'][num], "text" : wiki_data['text'][num]})
        for num in tqdm(range(len(wiki_data))):
            if num not in trash_ids:
                context = wiki_data['text'][num]
                try:
                    te = kss.split_chunks(context, max_length=1280,overlap=True)
                    for v in te:
                        es.index(index='document', body = {"title" : wiki_data['title'][num], "text" : v})
                except:
                    es.index(index='document', body = {"title" : wiki_data['title'][num], "text" : context})
                    continue
        
        if isinstance(query_or_dataset, str):
            # doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            query = {
                        'query':{
                            'bool':{
                                'must':[
                                        {'match':{'text':query_or_dataset}}
                                ],
                                'should':[
                                        {'match':{'text':query_or_dataset}}
                                ]
                            }
                        }
                    }

            doc = es.search(index='document',body=query,size=topk)['hits']['hits']
            print(len(doc))
            doc_scores=[]
            doc_contexts=[]
            for idx,i in enumerate(doc):
                doc_scores.append(i['_score'])
                doc_contexts.append(i['_source']['text'])
                print("Top-%d passage with score %.4f" % (idx + 1, i['_score']))
            es_server.kill()
            return doc_scores, doc_contexts
            

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            # with timer("query exhaustive search"):
            #     print("s")

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval with Elastic Search: ")):
            
                question=example['question']
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

                doc = es.search(index='document',body=query,size=topk)['hits']['hits']
                
                doc_scores=[]
                doc_contexts=[]
                doc_contexts_ids=[]
                for idx,i in enumerate(doc):
                    doc_scores.append(i['_score'])
                    doc_contexts.append(i['_source']['text'])
                    doc_contexts_ids.append(i['_id'])
                    
                
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_contexts_ids[0],
                    'context' : ' '.join([ctx for ctx in doc_contexts]) # 하나로 잇기
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            es_server.kill() # 서버 종료
            return cqas

    def indices_create(self,es):
        es.indices.create(index = 'document',
                  body = {
                      'settings':{
                          "number_of_replicas":2,
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

if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("./data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents.json"
    retriever = ElasticRetrieval(
        # tokenize_fn=tokenizer.tokenize,
        # tokenize_fn=tokenize,
        data_path="/opt/ml/code/data/",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    # retriever.get_sparse_embedding()

    print(f"query : {query}")
    with timer("single query by exhaustive search"):
        retriever.retrieve(query,100)
    # with timer("single query by faiss"):
    #     scores, indices = retriever.retrieve_faiss(query)
    


    # test bulk
    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retriever_with_elastic(full_ds, 100)
    #     df['correct'] = df['original_context'] == df['context']
    #     print("correct retrieval result by exhaustive search", df['correct'].sum() / len(df))

    print(indices)