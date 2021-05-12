from pororo import Pororo
from datasets import load_from_disk
from konlpy.tag import Kkma
import pickle

data_path = '/opt/ml/input/data/data/'
# datasets = load_from_disk(data_path)

with open('/opt/ml/input/data/data/wiki_1500_kss.pickle', 'rb') as file :
    p = pickle.load(file)
# p
print(len(p))
print(len(set(p)))

# train_dataset_context = datasets['train']['context']
# train_dataset_question = datasets['train']['question']
# print(train_dataset)


# print(Pororo.available_tasks())
# mrc = Pororo(task='mrc', lang='ko')
# pororo_tok = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko")
# print(pororo_tok)
# print(pororo_tok("어머니 항성에"))
# kkma = Kkma()
# print(kkma.pos("어머니 항성에"))

# def delete_josa(sentence) :
#     last_pos = kkma.pos(sentence)[-1]
#     if last_pos[1][0] == 'J' :
#         return sentence[:-len(last_pos[0])]
#     else :
#         return sentence

# print(delete_josa("어머니 항성에"))
# print(mrc(train_dataset_question[0], train_dataset_context[0]))

# print(pr)