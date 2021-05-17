#!/bin/bash

python inference.py --output_dir /opt/ml/models/roberta/inference \
		--model_name /opt/ml/models/roberta\
		--tokenizer_name xlm-roberta-large\
		--dataset_name /opt/ml/input/data/data/test_dataset \
		--do_predict \
		--retrieve_topk 20 \
		--overwrite_output_dir \
		--per_device_eval_batch_size 40
