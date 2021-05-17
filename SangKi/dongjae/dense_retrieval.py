from pororo import Pororo
from datasets import load_dataset, load_from_disk
from tqdm import tqdm, trange
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AdamW, TrainingArguments, get_linear_schedule_with_warmup, BertPreTrainedModel, BertModel, XLMRobertaModel, XLMRobertaTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from adamp import AdamP
from tokenization_kobert import KoBertTokenizer
import copy
import kss

def seed_everything(seed = 42) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# tokenizer = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko") 
dataset = load_dataset('squad_kor_v1')
# dataset = load_from_disk('../input/data/data/train_dataset')
print(dataset)

MODEL_NAME = 'monologg/kobert'
tokenizer = KoBertTokenizer.from_pretrained(MODEL_NAME)
model_max_length = tokenizer.model_max_length

class myEncoder(BertPreTrainedModel) :
    def __init__(self, config) :
        super(myEncoder, self).__init__(config)
        self.model = BertModel(config)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask = None, token_type_ids=None) :
        outputs = self.model(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        return outputs[1]

training_dataset = dataset['train']


temp = sorted(training_dataset['context'], key = lambda x : -len(x))[1]
input_1 = tokenizer(temp,
                    truncation=True,
                    max_length=model_max_length,
                    # stride=128,
                    return_overflowing_tokens=True,
                    padding="max_length"
                )
# print(temp)
temp_sentence = kss.split_sentences(temp, safe = True, max_recover_step=5, recover_step=5)
temp_indexs = [{'start':temp.index(current), 'end':temp.index(current) + len(current)} for current in temp_sentence]
print(kss.split_chunks(temp_sentence, overlap = True, indexes = temp_indexs))
# # print(input_1)
# sample_mapping = input_1.pop("overflow_to_sample_mapping")

# print(tokenizer.decode(input_1['input_ids']))
# print(tokenizer.decode(overflow_tokens))
# print(temp)
# print(tokenizer.decode(input_1['input_ids']))

# print(len(training_dataset['context']))
# print(len(set(training_dataset['context'])))

# q_seqs = tokenizer(training_dataset['question'], padding = 'max_length', truncation = True, return_tensors='pt')
# p_seqs = tokenizer(training_dataset['context'], padding = 'max_length', truncation = True, return_tensors='pt')

# train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
#                              q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

# validate_dataset = dataset['validation']
# q_seqs = tokenizer(validate_dataset['question'], padding='max_length', truncation=True, return_tensors='pt')
# p_seqs = tokenizer(validate_dataset['context'], padding='max_length', truncation=True, return_tensors='pt')

# valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
#                              q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])


# p_encoder = myEncoder.from_pretrained(MODEL_NAME)
# q_encoder = myEncoder.from_pretrained(MODEL_NAME)

# if torch.cuda.is_available():
#     p_encoder.cuda()
#     q_encoder.cuda()


# def train(args, train_dataset, valid_dataset, p_model, q_model):
  
#   # Dataloader
#     train_sampler = RandomSampler(train_dataset)
#     train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    
#     # Dataloader
#     valid_sampler = RandomSampler(valid_dataset)
#     valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.per_device_eval_batch_size)
    
#     # Optimizer
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params' : [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
#         {'params' : [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
#         {'params' : [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : args.weight_decay},
#         {'params' : [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0}
#     ]

#     t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
#     optimizer = AdamP(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

#     # Start training!
#     global_step = 0
#     best_acc = 0
#     best_step = 0
    
#     p_model.zero_grad()
#     q_model.zero_grad()

#     for _ in range(args.num_train_epochs):
#         train_losses = []
#         for batch in tqdm(train_loader):
#             q_model.train()
#             p_model.train()
            
#             batch = tuple(t.cuda() for t in batch)
                
#             p_inputs = {'input_ids' : batch[0],
#                        'attention_mask' : batch[1],
#                        'token_type_ids' : batch[2]
#                        }
            
#             q_inputs = {'input_ids' : batch[3],
#                        'attention_mask' : batch[4],
#                        'token_type_ids' : batch[5]
#                        }
            
#             p_outputs = p_model(**p_inputs)
#             q_outputs = q_model(**q_inputs)
            
#             sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
#             sim_scores = F.log_softmax(sim_scores, dim=1)
#             targets = torch.arange(0, sim_scores.shape[0]).long().cuda()
            
#             loss = F.nll_loss(sim_scores, targets)
#             train_losses.append(loss.item() / sim_scores.shape[0])
            
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             q_model.zero_grad()
#             p_model.zero_grad()
#             global_step += 1
            
#             if global_step % 1000 == 0:
#                 valid_losses = []
#                 valid_accs = []
#                 with torch.no_grad():
#                     p_model.eval()
#                     q_model.eval()
                    
#                     for batch in valid_loader:
#                         batch = tuple(t.cuda() for t in batch)
                        
#                         p_inputs = {'input_ids' : batch[0],
#                                     'attention_mask' : batch[1],
#                                     'token_type_ids' : batch[2]
#                                    }
            
#                         q_inputs = {'input_ids' : batch[3],
#                                     'attention_mask' : batch[4],
#                                     'token_type_ids' : batch[5]
#                                    }
                
#                         p_outputs = p_model(**p_inputs)
#                         q_outputs = q_model(**q_inputs)
                        
#                         sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
#                         sim_scores = F.log_softmax(sim_scores, dim=1)
#                         targets = torch.arange(0, sim_scores.shape[0]).long().cuda()
#                         predict = torch.argmax(sim_scores, dim=1).long()
                
#                         valid_loss = F.nll_loss(sim_scores, targets)      
#                         valid_acc = torch.mean((targets == predict).float())
                        
#                         valid_losses.append(valid_loss.item() / sim_scores.shape[0])
#                         valid_accs.append(valid_acc.item())
                        
#                 valid_acc = np.mean(valid_accs)
#                 print('train loss', np.mean(train_losses), 'valid loss', np.mean(valid_losses), 'valid acc', valid_acc)
#                 if valid_acc > best_acc:
#                     best_acc = valid_acc
#                     best_step = global_step
#                     best_p_model = copy.deepcopy(p_model)
#                     best_q_model = copy.deepcopy(q_model)
                    
#                 if global_step > best_step + 5000:
#                     return best_p_model, best_q_model
                    
#                 train_losses = []
                        
#             torch.cuda.empty_cache()
            
#     return best_p_model, best_q_model

# args = TrainingArguments(
#     output_dir="../results/dense_retireval",
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10, # For demonstration
#     weight_decay=1e-2,
#     save_total_limit=1,
# )

# p_encoder, q_encoder = train(args, train_dataset, valid_dataset, p_encoder, q_encoder)

# torch.save(p_encoder.state_dict(), './p_encoder.pt')
# torch.save(q_encoder.state_dict(), './q_encoder.pt')