#!/bin/bash

python inference.py --output_dir /opt/ml/models/roberta/inference \
		--model_name Dongjae/mrc2reader \
		--tokenizer_name xlm-roberta-large\
		--dataset_name /opt/ml/input/data/data/test_dataset \
		--do_predict \
		--retrieve_topk 30 \
		--overwrite_output_dir \
		--per_device_eval_batch_size 40
