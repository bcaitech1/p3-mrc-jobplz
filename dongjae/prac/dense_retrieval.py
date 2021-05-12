import json
import random
import numpy as np
from tqdm import tqdm, trange
import argparse
import copy

from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup, ElectraTokenizerFast, ElectraModel, ElectraPreTrainedModel
from datasets import load_from_disk, load_dataset

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# korquad dataset
dataset = load_dataset("squad_kor_v1")
print(dataset)

# baseline
model_checkpoint = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# custom
# model_checkpoint = 'kykim/electra-kor-base'
# tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint)

training_dataset = dataset['train']
q_seqs = tokenizer(training_dataset['question'], padding='max_length', truncation=True, return_tensors='pt')
p_seqs = tokenizer(training_dataset['context'], padding='max_length', truncation=True, return_tensors='pt')
train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                             q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

validate_dataset = dataset['validation']
q_seqs = tokenizer(validate_dataset['question'], padding='max_length', truncation=True, return_tensors='pt')
p_seqs = tokenizer(validate_dataset['context'], padding='max_length', truncation=True, return_tensors='pt')
valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                             q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)
        
        self.electra = ElectraModel(config)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        
        self.bert = BertModel(config)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

# baseline
p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()

# custom
# p_encoder = ElectraEncoder.from_pretrained(model_checkpoint).cuda()
# q_encoder = ElectraEncoder.from_pretrained(model_checkpoint).cuda()

args = TrainingArguments(
    output_dir='dense_retrieval',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=10,
    weight_decay=0.01
    )

def train(args, train_dataset, valid_dataset, p_model, q_model):
    
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    
    # Dataloader
    valid_sampler = RandomSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
        {'params' : [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
        {'params' : [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
        {'params' : [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    # start training
    global_step = 0
    best_acc = 0
    best_step = 0
    
    p_model.zero_grad()
    q_model.zero_grad()
    
    for _ in range(args.num_train_epochs):
        train_losses = []
        for batch in tqdm(train_loader):
            q_model.train()
            p_model.train()
            
            batch = tuple(t.cuda() for t in batch)
                
            p_inputs = {'input_ids' : batch[0],
                       'attention_mask' : batch[1],
                       'token_type_ids' : batch[2]
                       }
            
            q_inputs = {'input_ids' : batch[3],
                       'attention_mask' : batch[4],
                       'token_type_ids' : batch[5]
                       }
            
            p_outputs = p_model(**p_inputs)
            q_outputs = q_model(**q_inputs)
            
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            sim_scores = F.log_softmax(sim_scores, dim=1)
            targets = torch.arange(0, sim_scores.shape[0]).long().cuda()
            
            loss = F.nll_loss(sim_scores, targets)
            train_losses.append(loss.item() / sim_scores.shape[0])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1
            
            if global_step % 1000 == 0:
                valid_losses = []
                valid_accs = []
                with torch.no_grad():
                    p_model.eval()
                    q_model.eval()
                    
                    for batch in valid_loader:
                        batch = tuple(t.cuda() for t in batch)
                        
                        p_inputs = {'input_ids' : batch[0],
                                    'attention_mask' : batch[1],
                                    'token_type_ids' : batch[2]
                                   }
            
                        q_inputs = {'input_ids' : batch[3],
                                    'attention_mask' : batch[4],
                                    'token_type_ids' : batch[5]
                                   }
                
                        p_outputs = p_model(**p_inputs)
                        q_outputs = q_model(**q_inputs)
                        
                        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        targets = torch.arange(0, sim_scores.shape[0]).long().cuda()
                        predict = torch.argmax(sim_scores, dim=1).long()
                
                        valid_loss = F.nll_loss(sim_scores, targets)      
                        valid_acc = torch.mean((targets == predict).float())
                        
                        valid_losses.append(valid_loss.item() / sim_scores.shape[0])
                        valid_accs.append(valid_acc.item())
                        
                valid_acc = np.mean(valid_accs)
                print('train loss', np.mean(train_losses), 'valid loss', np.mean(valid_losses), 'valid acc', valid_acc)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_step = global_step
                    best_p_model = copy.deepcopy(p_model)
                    best_q_model = copy.deepcopy(q_model)
                    
                if global_step > best_step + 5000:
                    return best_p_model, best_q_model
                    
                train_losses = []
                        
            torch.cuda.empty_cache()
            
    return best_p_model, best_q_model

p_encoder, q_encoder = train(args, train_dataset, valid_dataset, p_encoder, q_encoder)
torch.save(p_encoder.state_dict(), './p_encoder.pt')
torch.save(q_encoder.state_dict(), './q_encoder.pt')

'''
valid_corpus = list(set([example['context'] for example in dataset['validation']]))
sample_idx = random.choice(range(len(dataset['validation'])))
query = dataset['validation'][sample_idx]['question']
ground_truth = dataset['validation'][sample_idx]['context']
print(query, ground_truth)

def to_cuda(batch):
    return tuple(t.cuda() for t in batch)

with torch.no_grad():
    p_encoder.eval()
    q_encoder.eval()
    
    q_seqs_val = tokenizer([query], padding='max_length', truncation=True, return_tensors='pt').to('cuda')
    q_emb = q_encoder(**q_seqs_val).to('cpu')
    
    p_embs = []
    for p in valid_corpus:
        p = tokenizer(p, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
        p_emb = p_encoder(**p).to('cpu').numpy()
        p_embs.append(p_emb)
        
    p_embs = torch.Tensor(p_embs).squeeze()
    
    print(p_embs.size(), q_emb.size())
    
dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
print(dot_prod_scores.size())

rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
print(dot_prod_scores)
print(rank)

k = 5
print('[Search query]\n', query, '\n')
print('[Ground truth passage]')
print(ground_truth, '\n')

for i in range(k):
    print('Top-%d pasage with score %.4f' % (i+1, dot_prod_scores.squeeze()[rank[i]]))
    print(valid_corpus[rank[i]])
'''