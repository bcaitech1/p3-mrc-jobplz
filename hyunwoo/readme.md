# hyunwoo
# train
## train.py
MRC 모델을 학습시키는 코드입니다.

이 코드를 이용해 MRC 모델을 학습시켰습니다.

다음과 같은 예시로 모델을 학습시킬 수 있습니다.
```bash
$ python train.py --output_dir /opt/ml/models/kobert-korquad\
		--dataset_name squad_kor_v1 \
		--model_name monologg/kobert \
		--do_train True \
		--do_eval True \
		--num_train_epoch 3 \
		--learning_rate 2e-5 \
		--per_device_train_batch_size 16 \
		--per_device_eval_batch_size 40 \
		--warmup_steps 500 \
		--logging_steps 100 \
		--save_total_limit 1 \
		--save_strategy epoch \
		--save_steps 2000 \
		--logging_dir /opt/ml/models/baseline/logs \
		--evaluation_strategy epoch \
		--fp16 True --fp16_backend amp --fp16_opt_level O2\
		--report_to wandb --run_name kobert-korquad-data \
```
이 포맷은 `train.sh` 파일에 저장되어 있습니다.

## DPR_train.py
Dense Passage Retrieval을 위한 모델을 학습시키는 코드입니다.

결과적으로는 성능이 너무 좋지 않아 이 코드로 학습시킨 모델은 사용하지 않았습니다.

다음과 같은 예시로 모델을 학습시킬 수 있습니다.
```bash
$ python DPR_train.py -name 2 -lr 2e-5
```

이 포맷은 `train.sh` 파일에 저장되어 있습니다.
# inference
## inference.py
학습시킨 모델로 inference를 수행하는 코드입니다.

retrieval을 포함하고 있으며 ODQA(Open Domain Question Answering)를 수행하는 코드입니다.

다음과 같은 예시로 inference를 수행합니다.
```bash
$ python inference.py --output_dir /opt/ml/models/roberta/inference \
		--model_name Dongjae/mrc2reader \
		--tokenizer_name xlm-roberta-large\
		--dataset_name /opt/ml/input/data/data/test_dataset \
		--do_predict \
		--retrieve_topk 30 \
		--overwrite_output_dir \
		--per_device_eval_batch_size 40
```
이 포맷은 `inference.sh` 파일에 저장되어 있습니다.

# retrieval
## retrieval.py
TF_IDF 알고리즘을 이용해 question과 가장 관련이 높은 문서를 반환합니다.

지정한 topk 만큼의 문서를 점수와 함께 데이터 프레임의 형태로 반환합니다.

다음과 같이 사용합니다.
```python3
from retrieval import SparseRetrieval

sparse_retriever = SparseRetrieval(tokenize_fn=tokenize,
                               data_path="/opt/ml/input/data/data",
                               context_path="wikipedia_documents.json")
# sparse_embedding
sparse_retriever.get_sparse_embedding()
df_sparse = sparse_retriever.retrieve(datasets['validation'], topk=data_args.retrieve_topk)
```
## DPR.py
`DPR_train.py`로 학습시킨 모델로 retrieval을 하기 위한 코드입니다.

하지만 학습에 실패하면서 실제로 사용하진 않았습니다.

## retrieval_dense.py
명수님이 구현하신 dense retrieval입니다.

명수님의 DPR 모델을 사용하는 것은 위의 `DPR.py`코드를 사용할 수도 있지만 

명수님이 구현하신 `utils_qa_ms.py`를 사용하기 위해 명수님의 코드를 사용하였습니다.

사용법은 다음과 같습니다.
```
from retrieval_dense import DenseRetrieval

dense_retriever = DenseRetrieval(p_path='thingsu/koDPR_context', q_path='thingsu/koDPR_question',
                                bert_path='kykim/bert-kor-base')
    
# dense_embedding
dense_retriever.get_dense_embedding()
df_dense = dense_retriever.retrieve(datasets['validation'], topk=data_args.retrieve_topk)
```

## retrieval_ms.py
명수님이 구현하신 sparse retrieval입니다.

역시 `utils_qa_ms.py`를 사용하기 위해 명수님의 코드를 사용하였습니다.

사용법은 `retrieval.py`와 같습니다.

# utils
## uiils_qa_ms.py
inference나 train에 필요한 util들이 구현되어 있는 코드입니다.

주로 최종 답안을 결정하거나 후처리를 하기 위해 이 코드를 많이 사용합니다.

명수님이 구현하신 이 util에 맞추기 위해 명수님의 retrieval 구현을 사용하였습니다.

# notebook
