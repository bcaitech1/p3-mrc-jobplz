"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
f
대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""


import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
import pickle

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.file_utils import SESSION_ID

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
from retrieval_dense import DenseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
import numpy as np
import pandas as pd 

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True
    training_args.fp16=True
    training_args.fp16_backend='amp'
    training_args.fp16_opt_level ='O2'
    
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config
    )

    # run passage retrieval if true
    if data_args.eval_retrieval:
        datasets_d = run_dense_retrieval(datasets, model_args, training_args, topk=100)
        datasets = run_sparse_retrieval(datasets, training_args, topk=100)
        with open('/opt/ml/input/data/data/dense.pickle', 'wb') as f :
            pickle.dump(datasets_d,f)
        with open('/opt/ml/input/data/data/sparse.pickle', 'wb') as f :
            pickle.dump(datasets_d,f)
        datasets = mix_dataset(dense = datasets_d, sparse = datasets, K = 0.52, cut = 20)
        datasets = dataset_formatting(datasets, training_args) 

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict :
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, topk=20)

def mix_dataset(dense, sparse, K = 1.2, cut = 30):
    CONTEXT = []
    CONTEXT_ID = []
    SCORES = []

    for (idx,d), (__,s) in zip(dense.iterrows(), sparse.iterrows()):
        CONTEXT.append(np.asarray(s['context']))
        CONTEXT_ID.append(np.asarray(s['context_id']))
        SCORES.append(np.asarray(s['scores']).copy())
        t_array = [c in s['context_id'] for c in d['context_id']]
        for k, c in enumerate(t_array):
            if c : # add score only
                ix = np.where(CONTEXT_ID[-1]==c)
                SCORES[-1][ix] = SCORES[-1][ix] + d['scores'][k] * K
            else :
                CONTEXT[-1] = np.append(CONTEXT[-1], d['context'][k])
                CONTEXT_ID[-1] = np.append(CONTEXT_ID[-1], d['context_id'][k])
                SCORES[-1] = np.append(SCORES[-1], d['scores'][k] * K)

        rank = SCORES[-1].argsort()
        CONTEXT_ID[-1] = CONTEXT_ID[-1][rank[::-1]][:cut]
        CONTEXT[-1] = CONTEXT[-1][rank[::-1]][:cut]
        SCORES[-1] = SCORES[-1][rank[::-1]][:cut]

    
    dataset = pd.DataFrame.from_dict({'context':CONTEXT, 'context_id':CONTEXT_ID, 
                            'id':dense['id'], 'question':dense['question'], 'scores':SCORES})

    return dataset

def run_sparse_retrieval(datasets, training_args,topk=20):
    #### retreival process ####
    wiki_path = "wiki_1280_kss.pickle"
    retriever = SparseRetrieval(
        tokenize_fn=tokenize,
        context_path=wiki_path)
    retriever.get_sparse_embedding()
    df = retriever.retrieve(datasets['validation'],topk=topk)

    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    return df

def dataset_formatting(datasets, training_args):
    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        f = Features({'context': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None),
                      'scores': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None),
                      "context_id" : Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
                      })

    elif training_args.do_eval: # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
        f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                                   'answer_start': Value(dtype='int32', id=None)},
                                          length=-1, id=None),
                      'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    dataset = DatasetDict({'validation': Dataset.from_pandas(datasets, features=f)})
    return dataset

def run_dense_retrieval(datasets, model_args, training_args,topk=20):
    #### retreival process ####
    wiki_path = "wiki_1280_kss.pickle"
    retriever = DenseRetrieval(p_path = model_args.p_path,
                               q_path = model_args.q_path,
                               bert_path = model_args.bert_path,
                                data_path="/opt/ml/input/data/data",
                                context_path=wiki_path)
    retriever.get_dense_embedding()
    df = retriever.retrieve(datasets['validation'],topk=topk)

    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    return df

def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, topk=5):
    # only for eval or predict
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)
    
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        context_length = [len(list(set(cs))) for cs in examples[context_column_name]]
        cumulative = [sum(context_length[:k]) for k, _ in enumerate(context_length)]
        question = [q for q, l in zip(examples[question_column_name],context_length) for _ in range(l)]
        context  = [c for cs in examples[context_column_name] for c in list(set(cs))]
        
        tokenized_examples = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples['ctx_rank'] = []
        tokenized_examples['scores'] = []
        
        on = 0
        cumulative.append(sum(context_length))
        # doc score 추가!
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            while cumulative[on+1] <= sample_mapping[i] :
                on += 1
            sample_index = on # sample_mapping[i] // topk
            rank_index = sample_mapping[i] - cumulative[on] # sample_mapping[i] % topk
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples['ctx_rank'].append(rank_index)
            tokenized_examples['scores'].append(examples["scores"][sample_index][rank_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,
        eval_examples=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - will create predictions.json
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=eval_dataset,
                                        test_examples=datasets['validation'])

        # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()
