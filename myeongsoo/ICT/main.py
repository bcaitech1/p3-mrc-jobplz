import os
import gc
import math
import random
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
import transformers
from transformers import AdamW
from transformers.models.auto.tokenization_auto import AutoTokenizer
from model import BertEncoder

import pickle
import util
import model as Model
import wandb
from konlpy.tag import Mecab

mecab = Mecab()
def tokenize_fn(text):
    # return text.split(" ")
    return mecab.morphs(text)

def check_answer(sim, contexts, answer, valid_rank=5) :
    "get accuracy based on given similarity score"
    sim = np.flip(np.argsort(sim, axis=1), axis=1)[:, :valid_rank]
    hits = []
    for a, s in zip(answer, sim) :
        hit = []
        for i in s :
            hit.append((a in contexts[i]))
        hits.append(hit)
    hits = np.array(hits)
    true_hit = np.zeros(hits.shape[0])!=0
    hit_rates = []
    for i in range(valid_rank) :
        true_hit = (hits[:, i].reshape(-1))|true_hit
        hit_rates.append(round((np.sum(true_hit)/len(true_hit))*100, 2))
        print("{} rank : {}".format(i+1, hit_rates[-1]))
    print('')
    return hit_rates[0], hit_rates


def main(config) :
    wandb.init(project="project-ICT", reinit=True)
    wandb.config.update(config)
    # prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_p)
    contexts, _, valid_qa = util.load_data(config, tokenizer)
    context_text = [context["clean_context"] for context in contexts]
    q_tokenized = [' '.join(qa["tokenized"]) for qa in valid_qa]
    q_wordpiece = [qa["wordpiece"] for qa in valid_qa]
    q_answer = [qa["answer"] for qa in valid_qa]

    vectorizer_path = '/opt/ml/Inverse-cloze-task-master/vectorizer_tfidf.bin'
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as file:
            tfidf = pickle.load(file)
    else :
        tfidf = TfidfVectorizer(tokenizer=tokenize_fn
                            , encoding="utf-8"
                            , ngram_range=(1, config.ngram))
        tfidf = tfidf.fit([' '.join(context["tokenized"]) for context in contexts])
        with open(vectorizer_path,"wb") as file:
            pickle.dump(tfidf,file)
    # define TF-IDF 
    print("TF-IDF Retrieval")
    context_path = '/opt/ml/Inverse-cloze-task-master/tfidf_context.bin'
    question_path = '/opt/ml/Inverse-cloze-task-master/tfidf_question.bin'
    if os.path.exists(context_path) and os.path.exists(question_path):
        with open(context_path, "rb") as file:
            tfidf_context = pickle.load(file)
        with open(question_path, "rb") as file:
            tfidf_question = pickle.load(file)
    else :
        tfidf_context = tfidf.transform([' '.join(context["tokenized"]) for context in contexts])
        with open(context_path, "wb") as file:
            pickle.dump(tfidf_context,file)
        tfidf_question = tfidf.transform(q_tokenized)
        with open(question_path, "wb") as file:
            pickle.dump(tfidf_question,file)
    tfidf_sim = util.get_sim(tfidf_question, tfidf_context)
    accuracy, rate = check_answer(tfidf_sim, context_text, q_answer)

    wandb.log({
            "tfidf accuracy": accuracy,
            "tfidf topk" : rate
            })
    del tfidf_context
    del tfidf_question
    gc.collect()

    # define ICT model
    if not config.use_cuda :
        _ = "cpu"
    vocab = dict()
    for k, v in tokenizer.vocab.items() :
        vocab[k] = v
    start_token = vocab["[CLS]"]
    p_encoder = BertEncoder.from_pretrained(config.bert_model_p)
    q_encoder = BertEncoder.from_pretrained(config.bert_model_q)
    
    wandb.watch(p_encoder)
    wandb.watch(q_encoder)

    if config.use_cuda :
        p_encoder.cuda()
        q_encoder.cuda()
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.01},
        {'params' : [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0},
        {'params' : [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.01},
        {'params' : [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay' : 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # make data loader
    def get_loader(data, batch_size) :
        data = TensorDataset(torch.from_numpy(data))
        return DataLoader(data
            , batch_size=batch_size
            , shuffle=True
            , sampler=None, drop_last=True)
    loader = get_loader(np.array([i for i in range(len(contexts))]), config.batch_size)

    def get_batch(index, contexts, start_token) :
        "make ICT batch data"
        sentence = [contexts[i]["sentence"] for i in index] # get sentences of paragraphs
        target_sentence = [random.randint(0, len(sen)-1) for sen in sentence] # set target sentence for ICT training
        remove_target = [random.random()<(1-config.remove_percent) for _ in range(len(target_sentence))] # determine removal of original sentence as mention in paper
        target_context = [sen[:i]+sen[i+remove:] for i, sen, remove in zip(target_sentence, sentence, remove_target)] # set sentences of target context
        target_context = [[y for x in context for y in x] for context in target_context] # concat sentences of context
        target_context = [[start_token]+context for context in target_context]
        target_sentence = [sen[i] for i, sen in zip(target_sentence, sentence)]
        target_sentence = [[start_token]+sen for sen in target_sentence]
        s, s_mask = util.pad_sequence(target_sentence, max_seq=config.max_seq, device='cuda') # pad sequence
        c, c_mask = util.pad_sequence(target_context, max_seq=config.max_seq, device='cuda')
        return s, s_mask, c, c_mask

    def save(model, epoch, accuracy) :
        "save model weight"
        model_to_save = model.module if hasattr(model,
                        'module') else model 
        save_dict = {
            'epoch' : epoch
            ,'accuracy' : accuracy
            ,'model': model_to_save.state_dict()
        }
        torch.save(save_dict, config.model_weight)

    def load(model, device) :
        "load model weight"
        model_to_load = model.module if hasattr(model,
                        'module') else model 
        load_dict = torch.load(config.model_weight
                                    , map_location=lambda storage
                                    , loc: storage.cuda(device))
        model_to_load.load_state_dict(load_dict['model'])
        return model_to_load

    def get_semantic_sim(p_encoder,q_encoder) :
        "make semantic embedding of context, question. and get similarity"
        context_embedding = []
        question_embedding = []
        p_encoder.eval()
        q_encoder.eval()
        with torch.no_grad() :
            for i in tqdm(range(0, len(contexts), config.test_batch_size)) :
                c = [[y for x in context["sentence"] for y in x] for context in contexts[i:i+config.test_batch_size]]
                c, c_mask = util.pad_sequence(c, max_seq=config.max_seq, device='cuda')
                c_encode = p_encoder(input_ids=c, attention_mask=c_mask)
                context_embedding.append(c_encode.detach().cpu().numpy())
            for i in tqdm(range(0, len(q_wordpiece), config.test_batch_size)) :
                q = [tokens for tokens in q_wordpiece[i:i+config.test_batch_size]]
                q, q_mask = util.pad_sequence(q, max_seq=config.max_seq, device='cuda')
                q_encode = q_encoder(input_ids=q, attention_mask=q_mask)
                question_embedding.append(q_encode.detach().cpu().numpy())
        context_embedding = np.concatenate(context_embedding, axis=0)
        question_embedding = np.concatenate(question_embedding, axis=0)
        return util.get_sim(question_embedding, context_embedding)  

    # train ICT model
    max_accuracy = -math.inf
    
    loss = nn.CrossEntropyLoss()

    print("ICT model Retrieval.")
    for e in range(config.epoch) :
        p_encoder.train()
        q_encoder.train()
        avg_loss = .0
        batch_num = len(loader)
        for step, batch in enumerate(tqdm(loader, total=batch_num)) :
            batch = batch[0]
            s, s_mask, c, c_mask = get_batch(batch, contexts, start_token)

            s_encode = q_encoder(input_ids=s, attention_mask=s_mask)
            c_encode = p_encoder(input_ids=c, attention_mask=c_mask)

            logit = torch.matmul(s_encode, c_encode.transpose(-2, -1))
            logit = F.log_softmax(logit, dim=1)
            target = torch.from_numpy(np.array([i for i in range(batch.size(0))])).long().to('cuda')
            loss_val = loss(logit, target).mean()
            # loss_val = F.nll_loss(logit, target)
            wandb.log({"Train Loss": loss_val})
            avg_loss += loss_val.item()
            loss_val.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            # q_encoder.zero_grad()
            # p_encoder.zero_grad()

            if step % 100 == 1 :
                print(f"Train Loss : {loss_val}, Step : {step}/{batch_num}")

        print("{} epoch, train loss : {}".format(e+1, round(avg_loss/batch_num, 2)))
        semantic_sim = get_semantic_sim(p_encoder,q_encoder)
        accuracy, rate = check_answer(semantic_sim, context_text, q_answer)

        wandb.log({"Epoch Loss": round(avg_loss/batch_num, 2),
                   "Epoch semantic_sim": semantic_sim,
                   "Epoch accuracy": accuracy,
                   "Epoch topk" : rate
                    })

        if accuracy > max_accuracy :
            max_accuracy = accuracy
            check_point = f"/opt/ml/Inverse-cloze-task-master/model/ICT_{e+1}_{accuracy:.2f}"
            p_encoder.save_pretrained(save_directory=f"{check_point}p")
            q_encoder.save_pretrained(save_directory=f"{check_point}q")


    # evaluate model with best performance weight
    p_encoder = p_encoder.from_pretrained(f"{check_point}p")
    q_encoder = q_encoder.from_pretrained(f"{check_point}q")
    
    if config.use_cuda :
        p_encoder.cuda()
        q_encoder.cuda()
        
    semantic_sim = get_semantic_sim(p_encoder, q_encoder)
    check_answer(semantic_sim, context_text, q_answer)

    # evalute ensemble 
    check_answer(semantic_sim*(1-config.sim_ratio)+tfidf_sim*config.sim_ratio
                    , context_text, q_answer)


if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", default="/opt/ml/Inverse-cloze-task-master/data.pkl"
                        , type=str, help="filename to save data")

    parser.add_argument("--ngram", default=2, type=int)
    parser.add_argument("--valid_rank", default=5, type=int)

    parser.add_argument("--do_lower", default=True, type=bool)
    parser.add_argument("--bert_model_p", default="kykim/bert-kor-base", type=str) #"CodeNinja1126/bert-p-encoder"
    parser.add_argument("--bert_model_q", default="kykim/bert-kor-base", type=str) #"CodeNinja1126/bert-q-encoder"

    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--max_seq", default=512, type=int)

    parser.add_argument("--epoch", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--remove_percent", default=2e-1, type=float)

    parser.add_argument("--sim_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)

    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--device", default='cuda')

    parser.add_argument("--devices"
                        , type=str
                        , default='cuda'
                        , help="gpu device ids to use concatend with '_' ex.'0_1_2_3'")

    main(parser.parse_args())