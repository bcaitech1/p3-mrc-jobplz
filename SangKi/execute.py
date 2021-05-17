import os
import subprocess
import train

# res = subprocess.check_output('ls -al', shell=True)
# print(type(res))

print(os.getcwd())
dir = os.getcwd()
print(os.listdir(os.getcwd()))

# argument_list runs수에 맞춰서 부여
runs = 3
project_name = "exp-01"
num_train_epochs_list = [3,2,2]
per_device_train_batch_size_list = [8,8,8]
learning_rate_list=[4e-06,4.5e-06,4e-05]
fp16_list=[True,True,True]
save_totel_limit_list = 1
save_steps_list = 100
output_dir_list = ["./models/train_dataset"]*runs
do_train_list = [True]*runs
do_eval_list = [False]*runs
fp16_backend_list = ['amp']*runs
model_name_or_path_list = ["deepset/xlm-roberta-large-squad2"]*runs
# deepset/xlm-roberta-large-squad2, monologg/kobert-lm,xlm-roberta-large, monologg/koelectra-base-v3-discriminator
overwrite_output_dir_list = [True]*runs
retrieval_list = ["sp"]*runs
p_model_list = ["deepset/xlm-roberta-large-squad2"]*runs # dense embedding 시 passage를 Encoding할 모델
q_model_list = ["deepset/xlm-roberta-large-squad2"]*runs # dense embedding 시 question을 Encoding할 모델
run_name_list = ["lr=4e-06","lr=4.5e-06","lr=5e-06"]

# inference_list 개수 맞춰서 부여

do_predict = True
do_eval = False

for i in range(1): # runs 만큼 모델 학습 및 추론 실험

    # train arguments
    train_args = {
    "num_train_epochs":num_train_epochs_list[i],
    "per_device_train_batch_size":per_device_train_batch_size_list[i],
    "learning_rate":learning_rate_list[i],
    "fp16":True,
    "save_total_limit" : 1,
    "save_steps" : 100,
    "output_dir" : f"./models/train_dataset-{model_name_or_path_list[i]}{i}" if model_name_or_path_list[i] is not None else f"./models/train_dataset",
    "do_train" : True,
    "do_eval" : False,
    "fp16_backend" : 'amp',
    "fp16_opt_level" :'O2',
    "model_name_or_path": model_name_or_path_list[i],
    "overwrite_output_dir":True,
    "use_wandb" : True, # whether you use wandb
    "project_name":f"xlm-roberta-large-squad2-ex", # wandb project name
    # "run_name":f"run-{i}",
    "run_name":run_name_list[i],
    "retrieval":retrieval_list[i] # 어떤 retrieval로 학습할 것인지(sp : sparse-embedding, de : dense-embedding)
    }
    del_embedding_file=True # 임베딩 파일 삭제할 것인지

    if retrieval_list[i]=='de':
        train_args["p_model"] = p_model_list[i]
        train_args["q_model"] = q_model_list[i]

    cmd = 'python3'+' '+'train.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in train_args.items() if v is not None])

    print(cmd)
    # 실행하는 부분

    if del_embedding_file: # 임베딩 bin파일을 삭제한다면
        if os.path.isfile(f"{dir}/data/sparse_embedding.bin"):
            os.remove(f"{dir}/data/sparse_embedding.bin")
        if os.path.isfile(f"{dir}/data/tfidv.bin"):
            os.remove(f"{dir}/data/tfidv.bin")

    # subprocess.run([cmd],shell=True)
    print("학습 끝")

    # 인퍼런스 평가

    model_dir = f"{dir}/models/train_dataset-{model_name_or_path_list[i]}{i}" if model_name_or_path_list[i] is not None \
        else f"{dir}/models/train_dataset"
    
    model_chk = [v for v in os.listdir(model_dir) if "checkpoint-" in v][-1] # 가장 나중에 만들어진 체크포인트
    do_eval,do_predict = True,False
    inf_args = {
        "output_dir":f"{dir}/outputs/predict-{i}", # eval을 낼 디렉토리
        "do_predict":do_predict,
        "do_eval": do_eval,
        "dataset_name":f"{dir}/data/test_dataset/" if do_predict else f"{dir}/data/train_dataset",
        "model_name_or_path":f"{dir}/models/train_dataset-{model_name_or_path_list[i]}{i}/{model_chk}",
        "overwrite_output_dir":True
    }
    do_eval,do_predict=do_predict,do_eval

    inference_cmd = 'python3'+' '+'inference.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in inf_args.items() if v is not None])
    print(inference_cmd)
    # subprocess.run([inference_cmd],shell=True)

    inf_args = {
        "output_dir":f"{dir}/outputs/predict-{1}", # predict를 낼 디렉토리
        "do_predict":do_predict,
        "do_eval":do_eval,
        "dataset_name":f"{dir}/data/test_dataset/" if do_predict else f"{dir}/data/train_dataset",
        "model_name_or_path":f"{dir}/models/train_dataset-{model_name_or_path_list[i]}{i}/{model_chk}",
        "overwrite_output_dir":True
    }

    inference_cmd = 'python3'+' '+'inference.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in inf_args.items() if v is not None])
    print(inference_cmd)
    # 인퍼런스 실행
    subprocess.run([inference_cmd],shell=True)
    print(f"{str(i)}번째 : eval or predict 끝")