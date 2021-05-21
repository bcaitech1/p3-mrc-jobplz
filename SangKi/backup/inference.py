"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""

import konlpy.tag as tag
import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    OtherArguments,
)

from es import *
from konlpy.tag import Mecab

from tqdm.notebook import tqdm

import copy

logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments,OtherArguments)
    )
    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()

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

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Load pretrained model and tokenizer
    
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    if model_args.model_name_or_path=="Dongjae/mrc2reader":
        tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
    else:
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
        print("SparseRetrieval Start")
        datasets = run_sparse_retrieval(datasets, training_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_sparse_retrieval(datasets, training_args):
    #### retreival process ####

    # retriever = SparseRetrieval(tokenize_fn=tokenize,
    #                             data_path="/opt/ml/code/dongjae/data",
    #                             context_path="wikipedia_documents.json")
    # retriever.get_sparse_embedding()
    # df = retriever.retrieve(datasets['validation'], 25)

    retriever = ElasticRetrieval(tokenize_fn=tokenize,
                                data_path="./data",
                                context_path="wikipedia_documents.json")
    # retriever.get_sparse_embedding()
    df = retriever.retrieve(datasets['validation'],topk=25)

    # print(df)
    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        # ㄱㅣ존
        f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})
        # 변경 
        # f = Features({'context': Value(dtype='string', id =None),
        #               'id': Value(dtype='string', id=None),
        #               'question': Value(dtype='string', id=None)})

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
    print("run_mrc gogo")
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
    # def accurate_prepare_validation_features(dataset):
    #     mecab = Mecab()
    #     input_ids = []
    #     attention_masks = []
    #     token_type_ids = []
    #     offset_mappings = []
    #     example_ids = []
    #     input_length = 512
    #     wrong_context = 0
    #     wrong_question = 0
    #     stride = 128
        
    #     for mrc_id in tqdm(range(len(dataset['context']))):
    #         is_wrong = False
                
    #         question = dataset['question'][mrc_id]
    #         mecab_question = mecab.morphs(question)
    #         offset_mapping = []
    #         bert_question = []
    #         index = 0
    #         for t in mecab_question:
    #             while index<len(question) and (question[index] == ' ' or question[index] == '\t' or question[index] == '\n'):
    #                 index += 1
    #             bert_t = tokenizer.tokenize(t)
    #             if '[UNK]' in bert_t:
    #                 bert_question.append('[UNK]')
    #                 offset_mapping.append((index, index+len(t)))
    #                 index += len(t)
    #             elif len(bert_t) == 0:
    #                 index += len(t)
    #             else:
    #                 if index > 0 and index<=len(question):
    #                     if question[index-1] != ' ' and question[index-1] != '\t' and question[index-1] != '\n':
    #                         bert_t[0] = '##' + bert_t[0]
    #                 if t[0] == '\u200e' or t[0] == '\xad':
    #                     index += 1
    #                 for bt in bert_t:
    #                     bert_question.append(bt)
    #                     if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
    #                         offset_mapping.append((index, index+len(bt)-2))
    #                         index += len(bt) - 2
    #                     else:
    #                         offset_mapping.append((index, index+len(bt)))
    #                         index += len(bt)
    #         for i, t in enumerate(bert_question):
    #             if t != '[UNK]':
    #                 if question[offset_mapping[i][0]:offset_mapping[i][1]] != t.replace("#", "", 2):
    #                     if question[offset_mapping[i][0]:offset_mapping[i][1]] != '#':
    #                         is_wrong = True
    #                         wrong_question += 1
    #                         # print(offset_mapping[i], question[offset_mapping[i][0]:offset_mapping[i][1]], t.replace("#", "", 2))
    #                         break
    #         # if is_wrong:
    #         #     print(dataset['question'][mrc_id])
    #         #     print(mecab_question)
    #         #     for t, o in zip(bert_question, offset_mapping):
    #         #         print(f'{t}:{o} ', end=' ')
    #         tokenized_question = tokenizer.convert_tokens_to_ids(bert_question)
    #         tokenized_question.insert(0, 101)
    #         tokenized_question.append(102)
        
    #         is_wrong = False
    #         context = dataset['context'][mrc_id]
    #         mecab_context = mecab.morphs(context)
    #         offset_mapping = []
    #         bert_context = []
    #         index = 0
    #         for t in mecab_context:
    #             while len(context)>index and (context[index] == ' ' or context[index] == '\t' or context[index] == '\n'):
    #                 index += 1
    #             bert_t = tokenizer.tokenize(t)
    #             if '[UNK]' in bert_t:
    #                 bert_context.append('[UNK]')
    #                 offset_mapping.append((index, index+len(t)))
    #                 index += len(t)
    #             elif len(bert_t) == 0:
    #                 index += len(t)
    #             else:
    #                 if index > 0 and len(context)>=index:
    #                     if context[index-1] != ' ' and context[index-1] != '\t' and context[index-1] != '\n':
    #                         bert_t[0] = '##' + bert_t[0]
    #                 if t[0] == '\u200e' or t[0] == '\xad':
    #                     index += 1
    #                 for bt in bert_t:
    #                     bert_context.append(bt)
    #                     if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
    #                         offset_mapping.append((index, index+len(bt)-2))
    #                         index += len(bt) - 2
    #                     else:
    #                         offset_mapping.append((index, index+len(bt)))
    #                         index += len(bt)
    #         for i, t in enumerate(bert_context):
    #             if t != '[UNK]':
    #                 if context[offset_mapping[i][0]:offset_mapping[i][1]] != t.replace("#", "", 2):
    #                     if context[offset_mapping[i][0]:offset_mapping[i][1]] != '#':
    #                         is_wrong = True
    #                         wrong_context += 1
    #                         # print(offset_mapping[i], context[offset_mapping[i][0]:offset_mapping[i][1]], t.replace("#", "", 2))
    #                         break
    #         # if is_wrong:
    #             # print(dataset['context'][mrc_id])
    #             # print(mecab_context)
    #             # for t, o in zip(bert_context, offset_mapping):
    #             #     print(f'{t}:{o} ', end=' ')
    #         tokenized_context = tokenizer.convert_tokens_to_ids(bert_context)
                        
    #         question_length = len(tokenized_question)
    #         context_length = len(tokenized_context)
    #         empty_length = input_length - question_length - 1
    #         index = 0
    #         while index < context_length:
    #             input_id = copy.deepcopy(tokenized_question)
    #             token_type_id = [0 for i in range(question_length)]
    #             example_offset_mapping = [None for i in range(question_length)]
    #             if index + empty_length > context_length:
    #                 input_id.extend(tokenized_context[index:context_length])
    #                 example_offset_mapping.extend(offset_mapping[index:context_length])
    #                 index = context_length
    #             else:
    #                 input_id.extend(tokenized_context[index:index+empty_length])
    #                 example_offset_mapping.extend(offset_mapping[index:index+empty_length])
    #                 index += empty_length - stride
                    
    #             input_id.append(102)
    #             attention_mask = [1 for i in range(len(input_id))]
    #             for i in range(question_length, len(input_id)):
    #                 token_type_id.append(1)
    #             for i in range(len(input_id), input_length):
    #                 input_id.append(0)
    #                 token_type_id.append(0)
    #                 attention_mask.append(0)
    #                 example_offset_mapping.append(None)
                        
    #             input_ids.append(input_id)
    #             attention_masks.append(attention_mask)
    #             token_type_ids.append(token_type_id)
    #             offset_mappings.append(example_offset_mapping)
    #             example_ids.append(dataset["id"][mrc_id])
                
    #     print('wrong_question', wrong_question, 'wrong_context', wrong_context)
    #     #  token_type_ids=token_type_ids, 
    #     return dict(input_ids=input_ids, attention_mask=attention_masks,offset_mapping=offset_mappings, example_id=example_ids)

    eval_dataset = datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        # accurate_prepare_validation_features,
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

    print(f"[training_args] : {training_args}")

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
