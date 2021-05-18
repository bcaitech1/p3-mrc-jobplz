import pandas as pd

# print(temp)
# valid_set = pd.read_csv('./valid_120_to_160.csv', sep = ',')
# predictions = pd.read_csv('./prediction.csv', sep = ',')

# print(valid_set)
# print(predictions)
# valid_set['pred'] = predictions
# count = 0 
# numbers = []
# for i, (gt, pred) in enumerate(zip(valid_set.answer, valid_set.pred)) :
#     if gt != pred :
#         print(f"{i+120}번째")
#         print(f"질문 : {valid_set.question[i]}")
#         print(f"정답 : {gt}")
#         print(f"예측값 : {pred}")
#         print(f"지문 : \n{valid_set.context[i]}")
#         count += 1
#         numbers.append(i+120)
# print(f"총 틀린 갯수 : {count}")
# print(f"틀린 문항 : {numbers}")