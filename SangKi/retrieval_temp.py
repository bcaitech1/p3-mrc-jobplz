import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
from utils_qa import BM25, tokenize
from rank_bm25 import BM25Plus
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="/opt/ml/input/data/data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # self.tokenized_contexts = [tokenize(context) for context in self.contexts]
        # self.bm25 = BM25Plus(self.contexts, tokenizer= tokenize_fn)

    def retrieve(self, query_or_dataset, topk=100):
        # assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                # doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
                tokenized_queries = [tokenize(query) for query in query_or_dataset['question']]
                bm25 = BM25Plus(self.contexts, tokenizer=tokenize, k1=1.0)
                doc_ranks = [bm25.get_top_n(tokenized_query, self.contexts, n=topk) for tokenized_query in tokenized_queries]
            
            # print(doc_indices)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    # "context_id": doc_indices[idx][0],  # retrieved id
                    'context_id' : 'temp',
                    # "context": self.contexts[doc_indices[idx][0]]  # retrieved doument            # 기본
                    'context' : ' '.join([doc_ranks[idx][i] for i in range(topk)]) # 하나로 잇기
                    # 'context' : [self.contexts[doc_indices[idx][i]] for i in range(topk)]
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            cqas.to_csv('./test.csv', sep = ',')
            # print(cqas)
            return cqas

    def get_relevant_doc(self, query, k=100):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]


if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("../input/data/data/train_dataset")
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
    retriever = SparseRetrieval(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="/opt/ml/input/data/data/",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    retriever.get_sparse_embedding()
    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query, 100)
    # with timer("single query by faiss"):
    #     scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds, 100)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search", df['correct'].sum() / len(df))
    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve_faiss(full_ds)
    #     df['correct'] = df['original_context'] == df['context']
    #     print("correct retrieval result by faiss", df['correct'].sum() / len(df))


