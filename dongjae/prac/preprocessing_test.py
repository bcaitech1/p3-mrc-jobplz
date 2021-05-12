from datasets import load_from_disk
import re
from code.train import prepare_validation_features

data_path = '/opt/ml/input/data/data/train_dataset'

dataset = load_from_disk(data_path)

print(dataset['train']['context'][-1])

def remove_newlines(example) :
    new_ones = [] 
    for i, context in enumerate(example) :
        new_one = re.sub(r'\\n+', ' ', context)
        new_ones.append(new_one)

    return new_ones
datasetz = remove_newlines(dataset['train']['context'])

def remove_double_space(example) :
    new_ones = []
    for i, context in enumerate(example) :
        new_one = re.sub('  ', ' ', context)
        new_ones.append(new_one)

    return new_ones
# print(remove_newlines(dataset['train']['context']))
print(remove_double_space(datasetz))

