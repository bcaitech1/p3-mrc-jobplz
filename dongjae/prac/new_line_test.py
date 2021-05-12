from datasets import load_from_disk, load_dataset
from transformers import BertTokenizer
import re

# 개행 치환
def remove_newlines(example) :
    new_one = re.sub(r'\\n+', ' ', example)

    return new_one
# 더블 스페이스 제거
def remove_double_space(example) :
    new_one = ' '.join(example.split())

    return new_one

# model_name = 'bert-base-multilingual-cased'
# tokenizer = BertTokenizer.from_pretrained(model_name)

# dataset = load_from_disk('/opt/ml/input/data/data/test_dataset')

# for v in dataset['validation']['question'] :
#     print(v)

# # dataset = load_dataset('squad_kor_v1')
# train_dataset = dataset['train']
# new_line_dataset = []

# # k = [v for v in train_dataset['answers'] if len(v['answer_start']) == 0]

# # print(k)

# for v in train_dataset : 
#     # print(v)
#     if '\\n' in v['context'] :
#         new_line_dataset.append(v['context'])

# print(f"총 train_dataset 개수 : {len(train_dataset['context'])}")
# print(f"new_line 포함 context 개수 : {len(new_line_dataset)}")

# # 개행문자 한개 \n 로 되어있는 데이터
# print(f"{'*' * 20} 개행문자 1개로 되어있는 데이터 {'*' * 20}")
# print(f"{'*' * 20} 원본 데이터 & 토크나이징 {'*' * 20}", end = '\n\n')
# original_target_1 = new_line_dataset[7]
# tokenized_original_target_1 = tokenizer.tokenize(original_target_1)

# print(original_target_1)
# print(tokenized_original_target_1)

# print(f"{'*' * 20} 전처리 후 데이터 & 토크나이징 {'*' * 20}", end = '\n\n')

# preprocess_target_1 = remove_double_space(remove_newlines(original_target_1))
# tokenized_preprocess_target_1 = tokenizer.tokenize(preprocess_target_1)

# print(preprocess_target_1)
# print(tokenized_preprocess_target_1)


# # 개행문자 두개 \n\n로 되어있는 데이터

# print(f"{'*' * 20} 개행문자 2개로 되어있는 데이터 {'*' * 20}")
# print(f"{'*' * 20} 원본 데이터 & 토크나이징 {'*' * 20}", end = '\n\n')
# original_target_2 = new_line_dataset[8]
# tokenized_original_target_2 = tokenizer.tokenize(original_target_2)

# print(original_target_2)
# print(tokenized_original_target_2)

# print(f"{'*' * 20} 전처리 후 데이터 & 토크나이징 {'*' * 20}", end = '\n\n')

# preprocess_target_2 = remove_double_space(remove_newlines(original_target_2))
# tokenized_preprocess_target_2 = tokenizer.tokenize(preprocess_target_2)

# print(preprocess_target_2)
# print(tokenized_preprocess_target_2)
