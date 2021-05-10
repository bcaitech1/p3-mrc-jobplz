import os
import subprocess
import train

# res = subprocess.check_output('ls -al', shell=True)
# print(type(res))

print(os.getcwd())
dir = os.getcwd()
print(os.listdir(os.getcwd()))

# argument_list 개수 맞춰서 부여
runs = 3
num_train_epochs_list = [8,6,10]
per_device_train_batch_size_list = [8,8,32]
learning_rate_list=[2e-5,2e-5,3e-5]
fp16_list=[True,True,True]
save_totel_limit_list = 1
save_steps_list = 100
output_dir_list = ["./models/train_dataset"]*runs
do_train_list = [True]*runs
do_eval_list = [False]*runs
fp16_backend_list = ['amp']*runs
model_name_or_path_list = ["monologg/kobert-lm","xlm-roberta-large","monologg/koelectra-base-v3-discriminator"]
overwrite_output_dir_list = [True]*runs

# inference_list 개수 맞춰서 부여

do_predict = True
do_eval = False

for i in range(runs): # runs 만큼 모델 학습 및 추론 실험

    # train arguments
    train_args = {
    "num_train_epochs":num_train_epochs_list[i],
    "per_device_train_batch_size":per_device_train_batch_size_list[i],
    "learning_rate":learning_rate_list[i],
    "fp16":True,
    "save_total_limit" : 1,
    "save_steps" : 100,
    "output_dir" : f"./models/train_dataset-{model_name_or_path_list[i]}" if model_name_or_path_list[i] is not None else f"./models/train_dataset",
    "do_train" : True,
    "do_eval" : False,
    "fp16_backend" : 'amp',
    "fp16_opt_level" :'O2',
    "model_name_or_path": model_name_or_path_list[i],
    "overwrite_output_dir":True
    }
    del_embedding_file=True # 임베딩 파일 삭제할 것인지
    cmd = 'python3'+' '+'train.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in train_args.items() if v is not None])

    print(cmd)
    # 실행하는 부분

    if del_embedding_file: # 임베딩 bin파일을 삭제할 것인지
        if os.path.isfile(f"{dir}/data/sparse_embedding.bin"):
            os.remove(f"{dir}/data/sparse_embedding.bin")
        if os.path.isfile(f"{dir}/data/tfidv.bin"):
            os.remove(f"{dir}/data/tfidv.bin")

    subprocess.run([cmd],shell=True)
    print("학습 끝")

    # 인퍼런스

    model_dir = f"{dir}/models/train_dataset-{model_name_or_path_list[i]}" if model_name_or_path_list[i] is not None \
        else f"{dir}/models/train_dataset"
    
    model_chk = [v for v in os.listdir(model_dir) if "checkpoint-" in v][-1] # 가장 나중에 만들어진 체크포인트
    do_eval,do_predict = True,False
    inf_args = {
        "output_dir":f"{dir}/outputs/predict-{i}", # eval을 낼 디렉토리
        "do_predict":do_predict,
        "do_eval": do_eval,
        "dataset_name":f"{dir}/data/test_dataset/" if do_predict else f"{dir}/data/train_dataset",
        "model_name_or_path":f"{dir}/models/train_dataset-{model_name_or_path_list[i]}/{model_chk}",
        "overwrite_output_dir":True
    }
    do_eval,do_predict=do_predict,do_eval

    inference_cmd = 'python3'+' '+'inference.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in inf_args.items() if v is not None])
    print(inference_cmd)
    subprocess.run([inference_cmd],shell=True)

    inf_args = {
        "output_dir":f"{dir}/outputs/predict-{i}", # predict를 낼 디렉토리
        "do_predict":do_predict,
        "do_eval":do_eval,
        "dataset_name":f"{dir}/data/test_dataset/" if do_predict else f"{dir}/data/train_dataset",
        "model_name_or_path":f"{dir}/models/train_dataset-{model_name_or_path_list[i]}/{model_chk}",
        "overwrite_output_dir":True
    }

    inference_cmd = 'python3'+' '+'inference.py'+' '+' '.join([f'--{k}={str(v)}' for k,v in inf_args.items() if v is not None])

    # 인퍼런스 실행
    subprocess.run([inference_cmd],shell=True)
    print(f"{str(i)}번째 : eval or predict 끝")