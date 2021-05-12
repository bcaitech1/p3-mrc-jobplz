import numpy as np
import pandas as pd
import json
import os
from tqdm.auto import tqdm

import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig
from datasets import Dataset, load_from_disk

from DPR_train import BertEncoder

class DenseRetrieval:
    def __init__(self, model_path='/opt/ml/models/DPR', data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # model load
        print('load models')
        model_config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
        # state_dict = torch.load(os.path.join(model_path, 'p_encoder.bin'))
        # self.p_encoder.load_state_dict(state_dict)
        # self.p_encoder = BertEncoder(model_config).cpu()
        self.p_encoder = BertEncoder.from_pretrained('/opt/ml/models/DPR_only_KLUE/p_encoder').cpu()
        # state_dict = torch.load(os.path.join(model_path, 'q_encoder.bin'))
        # self.q_encoder.load_state_dict(state_dict)
        # self.q_encoder = BertEncoder(model_config).cpu()
        self.q_encoder = BertEncoder.from_pretrained('/opt/ml/models/DPR_only_KLUE/q_encoder').cpu()
        torch.cuda.empty_cache()

        # tokenizer load
        print('load tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    def get_embedding(self):
        pickle_name = 'dense_embedding.bin'
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = []
            p_seqs = self.tokenizer(self.contexts, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
            p_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['token_type_ids'], p_seqs['attention_mask'])
            
            p_dataloader = DataLoader(p_dataset, 
                                        batch_size=40,
                                        num_workers=4
                                        )
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    self.p_encoder = self.p_encoder.cuda()
                    print('GPU enabled')

                self.p_encoder.eval()

                for batch in tqdm(p_dataloader):
                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)
                    
                    outputs = self.p_encoder(input_ids=batch[0],
                                        token_type_ids=batch[1],
                                        attention_mask=batch[2])
                    self.p_embedding.append(outputs.cpu().numpy())
                    torch.cuda.empty_cache()

            self.p_encoder = self.p_encoder.cpu()
            torch.cuda.empty_cache()

            len_vector = self.p_embedding[0].shape[-1]
            tmp_embedding = np.array(self.p_embedding[:-1]).reshape((-1, len_vector))
            self.p_embedding = np.concatenate((tmp_embedding, self.p_embedding[-1]), axis=0)

            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")
    

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "You must build dense embedding by self.get_embedding() before you run self.retrieve()."
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
            print('enter_elif')
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                for i, doc_id in enumerate(relev_doc_ids):
                    tmp = {
                        "question": example["question"],
                        "id": example['id'] + '-' + f'{i:02}',
                        "context_id": doc_id,  # retrieved id
                        "context": self.contexts[doc_id],  # retrieved doument
                        "score" :  doc_scores[idx][i]
                    }
                    if 'context' in example.keys() and 'answers' in example.keys():
                        tmp["original_context"] = example['context']  # original document
                        tmp["answers"] = example['answers']           # original answer
                    total.append(tmp)
            cqas = pd.DataFrame(total)
            return cqas

    
    def get_relevant_doc(self, query,  k=1):
        pass
        '''
        result = np.dot(query_vec , self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]
        '''
    
    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = []
        q_seqs = self.tokenizer(queries, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')

        q_dataset = TensorDataset(q_seqs['input_ids'], q_seqs['token_type_ids'], q_seqs['attention_mask'])
        
        q_dataloader = DataLoader(q_dataset, 
                                batch_size=40,
                                num_workers=4)
        
        with torch.no_grad():
            if torch.cuda.is_available():
                self.q_encoder = self.q_encoder.cuda()
                print('GPU enabled')

            self.q_encoder.eval()

            for batch in tqdm(q_dataloader):
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                
                outputs = self.q_encoder(input_ids=batch[0],
                                    token_type_ids=batch[1],
                                    attention_mask=batch[2])
                query_vec.append(outputs.cpu().numpy())
                torch.cuda.empty_cache()

        self.q_encoder = self.q_encoder.cpu()
        torch.cuda.empty_cache()

        len_vector = query_vec[0].shape[-1]
        tmp_embedding = np.array(query_vec[:-1]).reshape((-1, len_vector))
        query_vec = np.concatenate((tmp_embedding, query_vec[-1]), axis=0)
        print(query_vec.shape)

        result = np.dot(query_vec, self.p_embedding.T)
        print(result.shape)

        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices