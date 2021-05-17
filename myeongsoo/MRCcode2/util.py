import os
import re
import nltk
from nltk.stem.porter import PorterStemmer
import json
import pickle
import hashlib
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

stemmer = PorterStemmer()


def clean_text(text) :
    text = text.lower()
    clean_text = re.sub("[^0-9a-z]", "", text)
    if clean_text!="" :
        return clean_text
    else :
        return text


def sentence_tokenize(text) :
    "split paragraph to sentences"
    return nltk.sent_tokenize(text)


def word_tokenize(text) :
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(tok) for tok in tokens]
    return tokens


def parse_wiki_data(config, tokenizer, context_hash, contexts) :
    "parse paragraph, sentence, question, answer from squad-form datasets"
    qa = []
    with open("/opt/ml/input/data/data/wikipedia_documents.json", "r") as f:
        wiki = json.load(f)
    
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) 
    for context in tqdm(wiki_contexts) :
        # make hash value with clean text for checking duplicate paragraphs
        clean_context = clean_text(context)
        h = hashlib.md5(clean_context.encode("utf-8")).hexdigest()
        if h not in context_hash :
            context_hash.add(h)
            sentence = sentence_tokenize(context)
            # ICT requires paragraphs with at least 2 sentences for training!
            if len(sentence)<2 :
                continue
            contexts.append({
                "sentence":[tokenizer(sen)['input_ids'][1:-1] for sen in sentence]
                ,"tokenized":word_tokenize(context)
                ,"clean_context":clean_context # use for answer checking
            })


def parse_squad_data(config, filename, tokenizer) :
    "parse paragraph, sentence, question, answer from squad-form datasets"
    qa = []
    data = load_dataset('squad_kor_v1')
    data = data[filename]
    context_hash = set()
    contexts = []

    for d in tqdm(data) :
        context = d['context']
        # make hash value with clean text for checking duplicate paragraphs
        clean_context = clean_text(context)
        h = hashlib.md5(clean_context.encode("utf-8")).hexdigest()
        if h not in context_hash :
            context_hash.add(h)
            sentence = sentence_tokenize(context)
            # ICT requires paragraphs with at least 2 sentences for training!
            if len(sentence)<2 :
                continue
            contexts.append({
                # "sentence":[[tokenizer.vocab[tok] for tok in tokenizer.tokenize(sen)] for sen in sentence]
                "sentence":[tokenizer(sen)['input_ids'][1:-1] for sen in sentence]
                ,"tokenized":word_tokenize(context)
                ,"clean_context":clean_context # use for answer checking
            })
        question = d["question"]
        answer = clean_text(d['answers']['text'][0]) # make clean answer text for answer checking
        qa.append({
            "question":question
            ,"tokenized":word_tokenize(question)
            # ,"wordpiece":[tokenizer.vocab[tok] for tok in tokenizer.tokenize(question)]
            ,"wordpiece":tokenizer(question)['input_ids'][1:-1]
            ,"answer":answer
        })
    print(qa[0])
    return context, qa


def load_data(config, tokenizer) :
    "load datasets for training, validation"
    if os.path.exists('opt/ml/code/data.pkl') :
        with open('opt/ml/code/data.pkl', "rb") as handle :
            train_ctx, train_qa, valid_ctx, valid_qa = pickle.load(handle)
        return train_ctx, train_qa, valid_ctx, valid_qa

    train_ctx, train_qa = parse_squad_data(config, "train", tokenizer)
    valid_ctx, valid_qa = parse_squad_data(config, "validation", tokenizer)

    with open(config.data_file, "wb") as handle :
        pickle.dump((
            train_ctx, train_qa, valid_ctx, valid_qa
            ), handle)
    return train_ctx, train_qa, valid_ctx, valid_qa


def get_sim(v1, v2) :
    sim = cosine_similarity(v1, v2)
    return sim


def make_mask(x, pad_idx, decode=False):
    "Create a mask to hide padding and future words."
    mask = (x!=pad_idx)
    if decode :
        size = x.shape[-1]
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        mask = np.expand_dims(mask, axis=1) & (subsequent_mask == 0)
    return mask.astype('uint8')


def pad_sequence(x, max_seq=64, pad_idx=0, get_mask=True, decode=False, pad_max=False, device="cpu") :
    """
    padding given sequence with maximum length 
    generate padded sequence and mask
    """ 
    seq_len = np.array([min(len(seq), max_seq) for seq in x])
    if not pad_max :
        max_seq = max(seq_len)
    pad_seq = np.zeros((len(x), max_seq), dtype=np.int64)
    pad_seq.fill(pad_idx)
    for i, seq in enumerate(x):
        pad_seq[i, :seq_len[i]] = seq[:seq_len[i]]
    if get_mask :
        mask = make_mask(pad_seq, pad_idx, decode)
        mask = torch.from_numpy(mask).to(device)
    else :
        mask = None
    return torch.from_numpy(pad_seq).to(device), mask
