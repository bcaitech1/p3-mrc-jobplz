import torch
from tqdm.auto import tqdm
from tqdm import trange
import numpy as np
import os
import argparse
import re

import random
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    TensorDataset
)

from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import (
    XLMRobertaModel, AdamW, TrainingArguments, BertPreTrainedModel, BertModel,
    get_linear_schedule_with_warmup, RobertaModel
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

import wandb

from retrieval import SparseRetrieval
from utils_qa import tokenize

def main():
    parser = argparse.ArgumentParser(description='Set some train option')
    parser.add_argument('-lr', default=5e-5, type=float, help='learning rate (default : 5e-5)')
    parser.add_argument('-name', type=str, help='wandb_name')

    args = parser.parse_args()
    # wandb setting
    wandb.init(project='DPR_training', name='DPR_bert_with_TF_IDF' + args.name)
    wandb_config = wandb.config
    wandb_config.learning_rate = args.lr
    wandb_config.batch_szie = 8
    wandb_config.epochs = 20
    wandb_config.weigth_decay = 0.01

    # load dataset
    dataset_KLUE = load_from_disk('/opt/ml/input/data/data/train_dataset')
    dataset = load_dataset("squad_kor_v1")

    dataset_KLUE_train = dataset_KLUE['train'].map(features=dataset['train'].features, 
                                                    remove_columns=['document_id', '__index_level_0__'], 
                                                    keep_in_memory=True)
                                                       
    dataset_KLUE_valid = dataset_KLUE['validation'].map(features=dataset['validation'].features, 
                                                    remove_columns=['document_id', '__index_level_0__'], 
                                                    keep_in_memory=True)
    
    # dataset['train'] = concatenate_datasets([dataset_KLUE_train, dataset['train'].select(range(2000))])
    dataset['train'] = dataset_KLUE_train
    dataset['validation'] = concatenate_datasets([dataset_KLUE_valid, dataset['validation'].select(range(1000))])

    # TF-IDF retrieval
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data/data",
                                context_path="wikipedia_documents.json")
    retriever.get_sparse_embedding()
    df = retriever.retrieve(dataset['train'], topk=2)

    # model_name and load tokenizer
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize data
    training_dataset = dataset['train'] # .select(range(10))

    high_TF_IDF_list = []
    gold_list = []
    equal_num = 0
    for i in range(0, len(df), 2):
        cxt1 = re.sub(r'[ ]+',' ', df.loc[i]['context'])
        cxt1 = re.sub(r'\\n', '\n', cxt1)
        cxt2 = re.sub(r'[ ]+',' ', df.loc[i+1]['context'])
        cxt2 = re.sub(r'\\n', '\n', cxt2)
        origin_cxt = re.sub(r'[ ]+', ' ', df.loc[i]["original_context"])
        origin_cxt = re.sub(r'\\n', '\n', origin_cxt)
        
        gold_list.append(origin_cxt)
        if cxt1 == origin_cxt:
            equal_num += 1
            high_TF_IDF_list.append(cxt2)
        else:
            high_TF_IDF_list.append(cxt1)

    print(f'equal num: {equal_num}')
    
    q_seqs = tokenizer(training_dataset['question'], 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
    p_seqs = tokenizer(gold_list, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
    

    difficult_negative_seqs = tokenizer(high_TF_IDF_list, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
    

    # make dataset
    train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['token_type_ids'], p_seqs['attention_mask'],
                             q_seqs['input_ids'],  q_seqs['token_type_ids'], q_seqs['attention_mask'],
                             difficult_negative_seqs['input_ids'],  difficult_negative_seqs['token_type_ids'], 
                             difficult_negative_seqs['attention_mask'])

    # train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], 
    #                             q_seqs['input_ids'], q_seqs['attention_mask'])
    
    # make valid dataset
    validate_dataset = dataset['validation'] # .select(range(10))
    q_seqs = tokenizer(validate_dataset['question'], 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
    p_seqs = tokenizer(validate_dataset['context'], 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors='pt')
    valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['token_type_ids'], p_seqs['attention_mask'], 
                                 q_seqs['input_ids'], q_seqs['token_type_ids'], q_seqs['attention_mask'])
    # valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], 
    #                            q_seqs['input_ids'], q_seqs['attention_mask'])


    #load config
    config = AutoConfig.from_pretrained(model_name)

    # load model
    p_encoder = BertEncoder.from_pretrained(model_name, config=config)
    q_encoder = BertEncoder.from_pretrained(model_name, config=config)
    # p_encoder = RobertaEncoder.from_pretrained(model_name, config=config)
    # q_encoder = RobertaEncoder.from_pretrained(model_name, config=config)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
        print('GPU enabled')

    args = TrainingArguments(
        output_dir='/opt/ml/models/DPR_with_difficult',
        evaluation_strategy='epoch',
        learning_rate=args.lr,
        warmup_steps=100,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=40,
        num_train_epochs=20,
        weight_decay=0.1
    )

    # train
    q_encoder, p_encoder = train(args, train_dataset, valid_dataset, p_encoder, q_encoder)

    if os.path.isdir(args.output_dir) == False:
        os.makedirs(args.output_dir, exist_ok=True)

    # save model
    p_encoder.save_pretrained(os.path.join(args.output_dir, 'p_encoder'),save_config=True)
    q_encoder.save_pretrained(os.path.join(args.output_dir, 'q_encoder'), save_config=True)

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        
        self.bert = BertModel(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)
        
        self.roberta = XLMRobertaModel(config)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask)
        pooled_output = outputs[1]
        return pooled_output


def train(args, train_dataset, valid_dataset, p_model, q_model):
    
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size)
    # Optimizer
    '''
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if  any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},       
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    '''
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW([
                    {'params': p_model.parameters()},
                    {'params': q_model.parameters()}
                ], lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # wandb watch
    wandb.watch([p_model, q_model], criterion=optimizer)

    global_step = 0
    
    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange((int(args.num_train_epochs)), desc='Epoch')

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')

        p_model.train()
        q_model.train()

        for step, batch in enumerate(epoch_iterator):
            batch[0] = torch.cat((batch[0], batch[6]), dim=0)
            batch[1] = torch.cat((batch[1], batch[7]), dim=0)
            batch[2] = torch.cat((batch[2], batch[8]), dim=0)

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                
            p_inputs = {
                'input_ids' : batch[0],
                'token_type_ids' : batch[1],
                'attention_mask' : batch[2]
            }
            
            q_inputs = {
                'input_ids' : batch[3],
                'token_type_ids' : batch[4],
                'attention_mask' : batch[5]
            }
            
            p_outputs = p_model(**p_inputs)
            q_outputs = q_model(**q_inputs)
            
            # Calculate Similarity
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            
            # target
            targets = torch.arange(0, len(batch[3])).long()
            if torch.cuda.is_available():
                targets = targets.to('cuda')
            
            sim_scores = F.log_softmax(sim_scores, dim=1)
            
            loss = F.nll_loss(sim_scores, targets)
            wandb.log({'loss': loss, 'lr': optimizer.param_groups[0]['lr']})

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1
            
            torch.cuda.empty_cache()
            
        with torch.no_grad():
            # evaluation
            print('let\'s eval')

            p_model.eval()
            q_model.eval()

            p_outputs = []
            q_outputs = []
                
            for batch in tqdm(valid_loader):
                batch = tuple(t.cuda() for t in batch)
                    
                p_inputs = {'input_ids' : batch[0],
                            'token_type_ids' : batch[1],
                            'attention_mask' : batch[2]
                        }
            
                q_inputs = {'input_ids' : batch[3],
                            'token_type_ids' : batch[4],
                            'attention_mask' : batch[5]
                        }
                
                p_outputs.append(p_model(**p_inputs).cpu().numpy())
                q_outputs.append(q_model(**q_inputs).cpu().numpy())

            p_outputs = np.array(p_outputs).reshape((len(valid_dataset),-1))
            q_outputs = np.array(q_outputs).reshape((len(valid_dataset),-1))
                    
            sim_scores = np.dot(q_outputs, p_outputs.T)
            sorted_scores = np.argsort(sim_scores, axis=1)
                    
            top_1_score, top_5_score, top_10_score, top_20_score = 0, 0, 0, 0

            for idx in tqdm(range(len(valid_dataset))):
                if idx in sorted_scores[idx][:-2:-1]: top_1_score += 1
                if idx in sorted_scores[idx][:-6:-1]: top_5_score += 1
                if idx in sorted_scores[idx][:-11:-1]: top_10_score += 1
                if idx in sorted_scores[idx][:-21:-1]: top_20_score += 1

            top_1_score, top_5_score, top_10_score, top_20_score = top_1_score / len(valid_dataset), \
                                                                    top_5_score / len(valid_dataset), \
                                                                    top_10_score / len(valid_dataset), \
                                                                    top_20_score / len(valid_dataset) \
                    
            wandb.log({'acc/top_1': top_1_score, 'acc/top_5': top_5_score, 
                        'acc/top_10': top_10_score, 'acc/top_20': top_20_score})
            
    return q_model, p_model


if __name__ == '__main__':
    main()