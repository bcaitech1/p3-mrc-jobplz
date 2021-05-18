"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""
import collections
import json
import konlpy.tag as tag
import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from eunjeon import Mecab
from konlpy.tag import Kkma
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from typing import Optional, Dict, Tuple, Union
import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer
from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase
from pororo import Pororo
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


class PororoMrcFactory(PororoFactoryBase):

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]


    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}


    def load(self, device: str):

        if "brainbert" in self.config.n_model:
            #from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = (My_BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            # tagger = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko")
            tagger = tag.Kkma()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)

            
class My_BrainRobertaModel(RobertaModel):

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):

        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe64k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )

class BrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:

        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos(
        ) and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array(
                    [c
                     for c in s
                     if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c]
                      for c in s])
            for s in sentences
        ]

        if remove_bpe:
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2


            #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆
            # 여기 바꿨습니다 ☆
            #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆
            results = []
            # log_list.append(logits) # 디버깅용
            top_n = 5 # 수정하는 것으로 출력 개수를 달리할 수 있습니다. start logit 10개, end logit 10개로 -> 총 100개의 값을 제시합니다.
            
            starts = logits[:,0].argsort(descending = True)[:top_n].tolist() 

            for start in starts:
                ends = logits[:,1].argsort(descending = True).tolist()
                masked_ends = [end for end in ends if end >= start ]
                ends = (masked_ends+ends)[:top_n] # masked_ends가 비어있을 경우를 대비
                for end in ends:
                    answer_tokens = tokens[start:end + 1]
                    answer = ""
                    if len(answer_tokens) >= 1:
                        decoded = self.decode(answer_tokens)
                        if isinstance(decoded, str):
                            answer = decoded

                    score = (logits[:,0][start] + logits[:,1][end]).item()
                    results.append((answer, (start, end + 1), score ))
            
        return results

class PororoBertMrc(PororoBiencoderBase):

    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> Tuple[str, Tuple[int, int]]:

        postprocess = kwargs.get("postprocess", True)


        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆
        # 여기 바꿨습니다 ☆
        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆
        pair_results = self._model.predict_span(query, context)
        returns = []
        
        for pair_result in pair_results:
            span = self._callback(
            self._tagger,
            pair_result[0],
            ) if postprocess else pair_result[0]
            returns.append((span,pair_result[1],pair_result[2]))
        
        return returns



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
        config=config,
    )

    # run passage retrieval if true
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(datasets, training_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_sparse_retrieval(datasets, training_args):
    #### retreival process ####

    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data/data",
                                context_path="wikipedia_documents.json")
    retriever.get_sparse_embedding()
    df = retriever.retrieve(datasets['validation'], 25)

    # print(df)
    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        # ㄱㅣ존
        f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})
        # 현우님꺼
        # f = Features({'context': Value(dtype='string', id =None),
        #               'id': Value(dtype='string', id=None),
        #               'question': Value(dtype='string', id=None),
        #               'score' : Value(dtype='float32', id = None)})

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

    my_mrc_factory = PororoMrcFactory('mrc', 'ko', "brainbert.base.ko.korquad")
    my_mrc = my_mrc_factory.load(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    questions = datasets['validation'][question_column_name]
    contexts = datasets['validation'][context_column_name]
    ids = datasets['validation']['id']

    pos_tagger = tag.Kkma()

    def delete_josa(sentence) :
        pos = pos_tagger.pos(sentence)
        if len(pos) < 1 : return sentence
        last_pos = pos[-1]
        if last_pos[1][0] == 'J' :
            return sentence[:-len(last_pos[0])].strip()
        else :
            return sentence.strip()

    all_predictions = collections.OrderedDict()

    for i in tqdm(range(600)) :
        pred = my_mrc(questions[i], ' '.join(contexts[i].split()))
        answer = pred[0][0]
        all_predictions[ids[i]] = delete_josa(answer)
    

    print(f"Saving predictions.")
    with open(os.path.join('/opt/ml/results/predictions', '/pororo_mrc/predictions.json'), "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()
