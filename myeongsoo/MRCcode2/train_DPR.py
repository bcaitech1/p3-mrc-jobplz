from tqdm import trange
import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
import torch
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from tqdm.auto import tqdm
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    TensorDataset
)
import torch.nn.functional as F
from datasets import load_from_disk

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers import (
    AdamW, TrainingArguments, BertPreTrainedModel, BertModel,
    get_linear_schedule_with_warmup, RobertaModel
)
import pickle
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize, get_datasets
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
import numpy 
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
from pororo import Pororo
import random
import pandas as pd
import numpy as np

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        
        self.bert = BertModel(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

def get_adv_batch(index, dataset, toplist):
    adv = []
    for i in index :
        while True :
            x = toplist[int(i)][random.randint(0,19)] # 0~99모두 포함함
            if x != i :
                break
        adv.append(x)
    
    return dataset[adv]

def train(args, train_dataset, valid_dataset, adv_dataset, toplist, p_model, q_model):
    
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size)
    wiki_loader = DataLoader(adv_dataset, batch_size=args.per_device_train_batch_size)
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

            adv_batch = get_adv_batch(batch[6], adv_dataset, toplist)
            adv_batch = tuple(t.cuda() for t in adv_batch)

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                
            p_inputs = {'input_ids' : torch.cat([batch[0], adv_batch[0]], dim=0),
                       'attention_mask' : torch.cat([batch[1], adv_batch[1]], dim=0),
                       'token_type_ids' : torch.cat([batch[2], adv_batch[2]], dim=0)
                       }
            
            q_inputs = {'input_ids' : batch[3],
                       'attention_mask' : batch[4],
                       'token_type_ids' : batch[5]
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

            if global_step % 300 == 0 :
                print(f"global_step : {global_step}")
        if _ % 2 > 0 :
            with torch.no_grad():
                # evaluation
                print('let\'s eval')

                p_model.eval()
                q_model.eval()

                p_outputs = []
                q_outputs = []
                
                q_answers = []

                for batch in tqdm(valid_loader):
                        batch = tuple(t.cuda() for t in batch)
                    
                        q_inputs = {'input_ids' : batch[3],
                                    'token_type_ids' : batch[4],
                                    'attention_mask' : batch[5]
                                }
                        q_answers.append(batch[7].cpu().numpy())
                        q_outputs.append(q_model(**q_inputs).cpu().numpy())

                for batch in tqdm(wiki_loader):

                    batch = tuple(t.cuda() for t in batch)
                
                    p_inputs = {'input_ids' : batch[0],
                                'token_type_ids' : batch[1],
                                'attention_mask' : batch[2]
                            }  
                    p_outputs.append(p_model(**p_inputs).cpu().numpy())
                
                q_outputs = np.vstack(q_outputs)
                p_outputs = np.vstack(p_outputs)
                q_answers = np.hstack(q_answers)

                sim_scores = np.dot(q_outputs, p_outputs.T)
                sorted_scores = np.argsort(sim_scores, axis=1)
                # idx -> wiki_idx!!
                top_1_score, top_5_score, top_10_score, top_20_score, top_100_score = 0, 0, 0, 0, 0

                for k, idx in tqdm(enumerate(q_answers)):
                    if idx in sorted_scores[k][:-2:-1]: top_1_score += 1
                    if idx in sorted_scores[k][:-6:-1]: top_5_score += 1
                    if idx in sorted_scores[k][:-11:-1]: top_10_score += 1
                    if idx in sorted_scores[k][:-21:-1]: top_20_score += 1
                    if idx in sorted_scores[k][:-101:-1]: top_100_score += 1


                top_1_score, top_5_score, top_10_score, top_20_score, top_100_score =   top_1_score / len(valid_dataset), \
                                                                                        top_5_score / len(valid_dataset), \
                                                                                        top_10_score / len(valid_dataset), \
                                                                                        top_20_score / len(valid_dataset), \
                                                                                        top_100_score / len(valid_dataset) \

                wandb.log({'acc/top_1': top_1_score, 'acc/top_5': top_5_score, 
                            'acc/top_10': top_10_score, 'acc/top_20': top_20_score, 'acc/top_100': top_100_score})
                print('acc/top_1', top_1_score, 'acc/top_5', top_5_score, 
                            'acc/top_10', top_10_score, 'acc/top_20', top_20_score, 'acc/top_100', top_100_score)
                p_model.save_pretrained(os.path.join(args.output_dir, f'p_encoder_1_{_}'),save_config=True)
                q_model.save_pretrained(os.path.join(args.output_dir, f'q_encoder_1_{_}'), save_config=True)
            
    return q_model, p_model

def apply_kss(dataset):
    new_context = dataset['context']
    answer = dataset['answers']
    Q = dataset['question']
    contexts = []
    questions = []
    
    for i, (A, current) in tqdm(enumerate(zip(answer,new_context))) :
    # print(current)
        try :
            te = kss.split_chunks(current, max_length= 1280, overlap = True)
            for v in te :
                s, e = v.start, v.start + len(v.text) 
                if A['answer_start'][0] >=s and A['answer_start'][0] < e:
                    contexts.append(v.text)
                    questions.append(Q[i])
        except :
            contexts.append(current)
            questions.append(Q[i])
    
    return contexts, questions

import wandb

wandb.init(project="DenseRetrieval", reinit=True)
dataset = get_datasets("KLUE")
pororo_tok = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko")  # 제일 좋음 무조건 이거써야함 32보다 64가 조금 더 좋음    
def tokenize(text):
    # return text.split(" ")
    return pororo_tok(text)

wiki_path = "wiki_1280_kss.pickle"
retriever = SparseRetrieval(
    tokenize_fn=tokenize,
    context_path=wiki_path)
retriever.get_sparse_embedding()

# with open(os.path.join('/opt/ml/input/data/data', wiki_path), "rb") as f:
#     wiki = pickle.load(f)    
wiki = pd.read_csv('/opt/ml/input/data/data/wiki_cut.csv')

training_dataset = load_from_disk("/opt/ml/input/data/data/train_short")
toplist = retriever.retrieve(training_dataset, topk=20)['context_id']

model_name = 'kykim/bert-kor-base' # 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

#load config
config = AutoConfig.from_pretrained(model_name)

# load model
p_path = "/opt/ml/models/DPR_with_difficult/p_encoder_1_9"
q_path = "/opt/ml/models/DPR_with_difficult/q_encoder_1_9"
p_encoder = BertEncoder.from_pretrained(p_path)
q_encoder = BertEncoder.from_pretrained(q_path)

wiki_seq = tokenizer(list(wiki['context']),
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')
adv_dataset = TensorDataset(wiki_seq['input_ids'], wiki_seq['token_type_ids'], wiki_seq['attention_mask'])

questions, contexts = training_dataset['question'], training_dataset['context']
q_seqs = tokenizer(questions, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')

p_seqs = tokenizer(contexts, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['token_type_ids'], p_seqs['attention_mask'],
                        q_seqs['input_ids'],  q_seqs['token_type_ids'], q_seqs['attention_mask'], 
                        torch.LongTensor(list(range(len(p_seqs['input_ids'])))), torch.LongTensor(training_dataset['doc_id']))

validate_dataset = load_from_disk("/opt/ml/input/data/data/valid_short")

context, questions = validate_dataset['context'], validate_dataset['question']

q_seqs = tokenizer(questions, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')
p_seqs = tokenizer(context, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt')
valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['token_type_ids'], p_seqs['attention_mask'], 
                             q_seqs['input_ids'], q_seqs['token_type_ids'], q_seqs['attention_mask'],
                            torch.LongTensor(list(range(len(p_seqs['input_ids'])))), torch.LongTensor(validate_dataset['doc_id']))

if torch.cuda.is_available():
    p_encoder.cuda()
    q_encoder.cuda()
    print('GPU enabled')

args = TrainingArguments(
    output_dir='/opt/ml/models/DPR_with_difficult',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    warmup_steps=800,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=40,
    num_train_epochs=20,
    weight_decay=0.1
)

q_encoder, p_encoder = train(args, train_dataset, valid_dataset, adv_dataset, toplist, p_encoder, q_encoder)

if os.path.isdir(args.output_dir) == False:
    os.makedirs(args.output_dir, exist_ok=True)

# save model