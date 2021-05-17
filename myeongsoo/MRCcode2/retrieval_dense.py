from transformers import AutoTokenizer
import numpy as np
from tqdm import trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import pandas as pd
import multiprocessing
from contextlib import contextmanager
import time
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    TensorDataset
)
from transformers.utils.dummy_pt_objects import ElectraModel
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 
  
      outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
      
      pooled_output = outputs[1]

      return pooled_output

# class ElectraEncoder(ElectraPreTrainedModel):
#   def __init__(self, config):
#     super(ElectraEncoder, self).__init__(config)

#     self.electra = ElectraModel(config)
#     self.init_weights()
      
#   def forward(self, input_ids, 
#               attention_mask=None, token_type_ids=None): 
  
#       outputs = self.electra(input_ids,
#                           attention_mask=attention_mask,
#                           token_type_ids=token_type_ids)
      
#       pooled_output = outputs[0]

#       return pooled_output

class DenseRetrieval:
    def __init__(self, p_path, q_path, bert_path, data_path="/opt/ml/input/data/data/", context_path="wikipedia_documents.json"):
        # p_encoder, q_encoder, tokenizer
        self.data_path = data_path
        if context_path[-5:] == 'pickle':
            with open(os.path.join(data_path, context_path), "rb") as f:
                wiki = pickle.load(f)

            self.contexts = wiki # set 은 매번 순서가 바뀌므로
            print(f"Lengths of unique contexts : {len(self.contexts)}")
            self.ids = list(range(len(self.contexts)))
        else :
            wiki = pd.read_csv(os.path.join(data_path, context_path))
            self.contexts = list(wiki['context'])
            self.ids = list(wiki['doc_id'])

        self.p_path = p_path
        self.q_path = q_path
        self.p_encoder = BertEncoder.from_pretrained(self.p_path)
        self.q_encoder = BertEncoder.from_pretrained(self.q_path)

        self.bert_path = bert_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
    
    def get_dense_embedding(self):
        
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path) :
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            self.p_embedding = self.build_embedding()
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_embedding(self):
        with torch.no_grad():
            self.p_encoder.eval()
            wiki_embs = []
            wiki_loader = DataLoader(self.contexts, batch_size=8)
            for p in tqdm(wiki_loader):
                p_inputs = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                wiki_emb = self.p_encoder(**p_inputs).to('cpu').numpy()
                wiki_embs.append(wiki_emb)

            wiki_embs = np.vstack(wiki_embs)  # (num_passage, emb_dim)wiki_embs = np.vstack(wiki_embs)
            wiki_embs = torch.FloatTensor(wiki_embs)
        return wiki_embs


    def retrieve(self, query_or_dataset, topk=5):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                def softmax(x):
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()
                # doc_scores = softmax(doc_scores)
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx],  # retrieved id
                    "context" : [self.contexts[i] for i in doc_indices[idx]],
                                # self.contexts[doc_indices[idx][0]]   retrieved doument
                    "scores" : softmax([x/200 for x in doc_scores[idx]])
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                    tmp["id"] = example['doc_id']
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=5):
        with timer("transform"):
            with torch.no_grad(): # 터질 수 있음
                self.q_encoder.eval()
                q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_embs = self.q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

        with timer("query ex search"):
            result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))
        if not isinstance(result, np.ndarray):
            result = result.numpy()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=5):
        with timer("transform"):
            with torch.no_grad():
                self.q_encoder.eval()
                q_embs = []
                valid_loader = DataLoader(queries, batch_size=20)
                for batch in tqdm(valid_loader):
                    q_seqs = self.tokenizer(batch, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    q_embs.append(self.q_encoder(**q_seqs).to('cpu').numpy())
        q_embs = np.vstack(q_embs)
        q_embs = torch.FloatTensor(q_embs)
        result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))
        if not isinstance(result, np.ndarray):
            result = result.numpy()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices