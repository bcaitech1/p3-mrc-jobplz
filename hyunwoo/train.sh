#!/bin/bash

python train.py --output_dir /opt/ml/models/kobert-korquad\
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
	 
