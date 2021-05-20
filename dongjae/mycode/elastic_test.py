import os
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
import json
import pandas as pd
from tqdm import tqdm
import kss
from datasets import load_from_disk, Sequence, Value, Features, Dataset, DatasetDict

import konlpy.tag as tag
import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
import re
import pickle

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


es = Elasticsearch('localhost:9200')

dataset = load_from_disk('/opt/ml/input/data/data/test_dataset')

dataset = dataset['validation']
# # 기본
dataset_question = dataset['question']
# print(dataset_question)
dataset_id = dataset['id']
dataset_context = []

# ES 진행
def elastic_search(topk, post_type) :
    # single - 하나로 이어서
    if not post_type :
        print(f'start single -- {topk} document')
        for question in tqdm(dataset_question) :
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

            docs = es.search(index='document_original',body=query,size=topk)['hits']['hits']
            doc = '\n\n'.join([v['_source']['text'] for v in docs]) 
            dataset_context.append(doc)

        df = pd.DataFrame({'id' : dataset_id, 'question' : dataset_question, 'context' : dataset_context})

        f = Features({'context': Value(dtype='string', id=None),
                        'id': Value(dtype='string', id=None),
                        'question': Value(dtype='string', id=None)})
        
        datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
        return datasets
    # score
    else :
        dataset_score = []
        dataset_context_id = []
        print(f'start score documents -- {topk} document')
        for i, question in tqdm(enumerate(dataset_question)) :
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

            docs = es.search(index='document_original_set',body=query,size=topk)['hits']['hits']
            dataset_context.append([doc['_source']['text'] for doc in docs])
            dataset_score.append([doc['_score'] for doc in docs])
            dataset_context_id.append([doc['_source']['title'] for doc in docs])

        df = pd.DataFrame({'question' : dataset_question, 'id' : dataset_id, 'context_id' : dataset_context_id, 'context' : dataset_context, 'scores' : dataset_score})

        return df

# topk
topk = int(input('topk -> '))
post_type = int(input('type 0(single) / 1(score) -> '))

# 데이터셋 생성
print('make new dataset!!!')
datasets = elastic_search(topk, post_type)

data_path = '/opt/ml/input/data/data'
topk100_set_pickle = 'topk100_set.pickle'

with open(os.path.join(data_path, topk100_set_pickle), "wb") as file:
    pickle.dump(datasets, file)