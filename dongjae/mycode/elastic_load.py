import pickle
import os


data_path = '/opt/ml/input/data/data'
topk100_pickle = 'topk100.pickle'
with open(os.path.join(data_path, topk100_pickle), 'rb') as file :
    contexts = pickle.load(file)

print(contexts.head())