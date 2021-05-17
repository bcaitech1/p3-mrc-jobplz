#!/bin/bash

python inference.py --output_dir /opt/ml/models/roberta/inference \
		--model_name Dongjae/mrc2reader\
		--tokenizer_name deepset/xlm-roberta-large-squad2\
		--dataset_name /opt/ml/input/data/data/test_dataset \
		--do_predict \
		--retrieve_topk 20 \
		--overwrite_output_dir \
		--per_device_eval_batch_size 40
