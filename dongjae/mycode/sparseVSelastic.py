import os
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
import json
import pandas as pd
from tqdm import tqdm
import kss
import re
from datasets import load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from retrieval_temp import SparseRetrieval
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from datasets import load_metric, load_from_disk, load_dataset, concatenate_datasets
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

# datasets = load_from_disk('/opt/ml/input/data/data/train_dataset')
datasets = load_dataset('squad_kor_v1')
# elastic search load
es = Elasticsearch('localhost:9200')

dataset = datasets['validation']
# # 기본
dataset_question = dataset['question']
# print(dataset_question)
dataset_id = dataset['id']
dataset_context = []

# ES 진행
def elastic_search(topk) :
    # single - 하나로 이어서
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
        dataset_context.append([v['_source']['text'] for v in docs])

    df = pd.DataFrame({'id' : dataset_id, 'question' : dataset_question, 'context' : dataset_context})

    f = Features({'context': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
                    'id': Value(dtype='string', id=None),
                    'question': Value(dtype='string', id=None)})
        
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    return datasets

# topk
topk = int(input('topk -> '))

# 데이터셋 생성

print('*' * 20, ' 엘라스틱 데이터셋 생성 시작 ', '*' * 20)
elastic_datasets = elastic_search(topk)

print('*' * 20, ' 엘라스틱 데이터셋 생성 완료 ', '*' * 20)
# sparse embedding load
# Set seed before initializing model.
set_seed(42)

print('*' * 20, ' Sparse 데이터셋 생성 시작 ', '*' * 20)
retriever = SparseRetrieval(tokenize_fn=tokenize,
                            data_path="/opt/ml/input/data/data",
                            context_path="wikipedia_documents.json")
retriever.get_sparse_embedding()
df = retriever.retrieve(datasets['validation'], topk)

print('*' * 20, ' Sparse 데이터셋 생성 완료 ', '*' * 20)
'''
    Ground Truth
'''
# wiki_path = '/opt/ml/input/data/data/ground_truth.json'
# with open(wiki_path) as wiki :
#     wiki_data = json.load(wiki)
# ground_truth = [v.split(':')[2:][:10] for v in wiki_data.keys()]
ground_truth = dataset['context']

print(elastic_datasets)

elastic_contexts = elastic_datasets['validation']['context']      # list
sparse_contexts = df['context']                     # list

data_len = len(elastic_contexts)
document_same_list = []

elastic_point = 0
sparse_point = 0

elastic_rank = []
sparse_rank = []
print('*' * 20, ' 비교 시작 ', '*' * 20)

for i, (ela, spa) in tqdm(enumerate(zip(elastic_contexts, sparse_contexts))) :
    temp_same = 0
    temp_elastic_point = False
    temp_sparse_point = False
    for ela_idx, temp_ela in enumerate(ela) :
        if temp_ela in spa :
            temp_same += 1
        if (temp_ela == ground_truth[i]) and (not temp_elastic_point) :
            temp_elastic_point = True
            elastic_rank.append(ela_idx+1)
            elastic_point += 1

    document_same_list.append(temp_same / topk)

    for spa_idx, temp_spa in enumerate(spa) :
        if (temp_spa == ground_truth[i]) and (not temp_sparse_point) :
            temp_sparse_point = True
            sparse_rank.append(spa_idx+1)
            sparse_point += 1


print('*' * 20, ' 비교 완료 ', '*' * 20, end = '\n\n')
print('*' * 20, 'topk : ', topk, ' 결과 ', '*' * 20)
print(f'document_same 평균 : {sum(document_same_list) / data_len}')
print(f'document_same 최소 : {min(document_same_list)}')
print(f'document_same 최대 : {max(document_same_list)}', end = '\n\n')
print(f'elastic ground truth 평균 : {elastic_point / data_len}')
print(f'sparse ground truth 평균 : {sparse_point / data_len}', end = '\n\n')

print(f'elastic ground truth rank 평균 : {sum(elastic_rank) / data_len}')
print(f'elastic ground truth rank 최대 : {max(elastic_rank)}')
print(f'elastic ground truth rank 최소 : {min(elastic_rank)}')
print(f'elastic ground truth top5 accuracy : {len([v for v in elastic_rank if v <= 5]) / data_len}')
print(f'elastic ground truth top10 accuracy : {len([v for v in elastic_rank if v <= 10]) / data_len}')
print(f'elastic ground truth top15 accuracy : {len([v for v in elastic_rank if v <= 15]) / data_len}')
print(f'elastic ground truth top20 accuracy : {len([v for v in elastic_rank if v <= 20]) / data_len}')
print(f'elastic ground truth top25 accuracy : {len([v for v in elastic_rank if v <= 25]) / data_len}')
print(f'elastic ground truth top30 accuracy : {len([v for v in elastic_rank if v <= 30]) / data_len}')
print(f'elastic ground truth top40 accuracy : {len([v for v in elastic_rank if v <= 40]) / data_len}')
print(f'elastic ground truth top50 accuracy : {len([v for v in elastic_rank if v <= 50]) / data_len}', end = '\n\n')


print(f'sparse ground truth rank 평균 : {sum(sparse_rank) / data_len}')
print(f'sparse ground truth rank 최대 : {max(sparse_rank)}')
print(f'sparse ground truth rank 최소 : {min(sparse_rank)}')
print(f'sparse ground truth top5 accuracy : {len([v for v in sparse_rank if v <= 5]) / data_len}')
print(f'sparse ground truth top10 accuracy : {len([v for v in sparse_rank if v <= 10]) / data_len}')
print(f'sparse ground truth top15 accuracy : {len([v for v in sparse_rank if v <= 15]) / data_len}')
print(f'sparse ground truth top20 accuracy : {len([v for v in sparse_rank if v <= 20]) / data_len}')
print(f'sparse ground truth top25 accuracy : {len([v for v in sparse_rank if v <= 25]) / data_len}')
print(f'sparse ground truth top30 accuracy : {len([v for v in sparse_rank if v <= 30]) / data_len}')
print(f'sparse ground truth top40 accuracy : {len([v for v in sparse_rank if v <= 40]) / data_len}')
print(f'sparse ground truth top50 accuracy : {len([v for v in sparse_rank if v <= 50]) / data_len}', end = '\n\n')