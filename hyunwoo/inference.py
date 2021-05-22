"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""


import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from collections import defaultdict
import re

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa_ms import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval_ms import SparseRetrieval
from retrieval_dense import DenseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

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

    if os.path.isdir(data_args.dataset_name):
        datasets = load_from_disk(data_args.dataset_name)
    else:
        datasets = load_dataset(data_args.dataset_name)
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
        config=config,
    )

    # run passage retrieval if true

    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(datasets, training_args, data_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(datasets, training_args, data_args):
    #### retreival process ####
    # test code
    datasets['validation'] = datasets['validation']
    
    # sparse Retrieval
    sparse_retriever = SparseRetrieval(tokenize_fn=tokenize,
                               data_path="/opt/ml/input/data/data",
                               context_path="wikipedia_documents.json")
    #
    # sparse_embedding
    sparse_retriever.get_sparse_embedding()
    df_sparse = sparse_retriever.retrieve(datasets['validation'], topk=data_args.retrieve_topk)

    # dense Retrieval
    dense_retriever = DenseRetrieval(p_path='thingsu/koDPR_context', q_path='thingsu/koDPR_question',
                                bert_path='kykim/bert-kor-base')
    
    # dense_embedding
    dense_retriever.get_dense_embedding()
    df_dense = dense_retriever.retrieve(datasets['validation'], topk=data_args.retrieve_topk)

    # merging_embeddings
    df = merging_retrieval(df_sparse, df_dense, dense_retriever.contexts, topk=data_args.retrieve_topk)

    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        '''
        f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})
                      # 'score': Value(dtype='float32', id=None)})
        '''

        # ms_style_feature
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

    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    # only for eval or predict
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)

    def prepare_validation_features_ms(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        context_length = [len(cs) for cs in examples[context_column_name]]
        cumulative = [sum(context_length[:k]) for k, _ in enumerate(context_length)]
        question = [q for q, l in zip(examples[question_column_name],context_length) for _ in range(l)]
        context  = [c for cs in examples[context_column_name] for c in cs]
        
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

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
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

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

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
        prepare_validation_features_ms,
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
            output_dir=training_args.output_dir
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


def merging_retrieval(df_sparse, df_dense, contexts, topk=30):
    k = 1.1
    
    dict_context_list = []
    for idx in range(len(df_sparse)):
        dict_context = defaultdict(float)
        data = df_sparse.loc[idx]
        for context_id, score in zip(data['context_id'], data['scores']):
            dict_context[context_id] = score
        dict_context_list.append(dict_context)

    for idx, dict_context in enumerate(dict_context_list):
        data = df_dense.loc[idx]
        for context_id, score in zip(data['context_id'], data['scores']):
            dict_context[context_id] += k * score

    context_score_pair_list = []
    for dict_context in dict_context_list:
        tmp_list = list(dict_context.items())
        tmp_list.sort(key=lambda x : x[1], reverse=True)
        context_score_pair_list.append(tmp_list)
    
    for idx in range(len(df_sparse)):
        tmp_id_list = [cxt_id[0] for cxt_id in context_score_pair_list[idx]]
        merging_cxt = []
        for cxt_id in tmp_id_list[:topk]:
            tmp_context = contexts[cxt_id]
            tmp_context = re.sub(r'\\n','\n', tmp_context) 
            tmp_context = re.sub(r'( )+',' ', tmp_context)
            merging_cxt.append(tmp_context)
        df_sparse.loc[idx]['context'] = merging_cxt
        df_sparse.loc[idx]['context_id'] = tmp_id_list[:topk]
        df_sparse.loc[idx]['socre'] = [cxt_id[1] for cxt_id in context_score_pair_list[idx]][:topk]
    
    return df_sparse

if __name__ == "__main__":
    main()
