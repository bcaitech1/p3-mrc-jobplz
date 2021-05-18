#######################################################################################################################
# train_dataset.map(prepare_train_features -> accurate_prepare_train_features로 변경해서 train_dataset 생성)
# *주의* map 옵션에 num_procs = 12 사용하면 안 됨
# klue + korquad tokenize할 때 대략 25분 소요되기 때문에
# 생성한 train_dataset을 bin 파일로 저장하고 load 해서 사용하는 것을 추천한다.
#######################################################################################################################

def accurate_prepare_train_features(dataset):
    mecab = Mecab()
    input_ids = []
    attention_masks = []
    token_type_ids = []
    start_positions = []
    end_positions = []
    input_length = 512
    wrong_answer = 0
    wrong_tokenize = 0
    stride = 128
    
    for mrc_id in range(len(dataset['answers'])):
        is_wrong = False
        answer = dataset['answers'][mrc_id]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        question = dataset['question'][mrc_id]
        mecab_question = mecab.morphs(question)
        offsets = []
        bert_question = []
        index = 0
        for t in mecab_question:
            while question[index] == ' ' or question[index] == '\t' or question[index] == '\n':
                index += 1
            bert_t = tokenizer.tokenize(t)
            if '[UNK]' in bert_t:
                bert_question.append('[UNK]')
                offsets.append((index, index+len(t)))
                index += len(t)
            elif len(bert_t) == 0:
                index += len(t)
            else:
                if index > 0:
                    if question[index-1] != ' ' and question[index-1] != '\t' and question[index-1] != '\n':
                        bert_t[0] = '##' + bert_t[0]
                for bt in bert_t:
                    bert_question.append(bt)
                    if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
                        offsets.append((index, index+len(bt)-2))
                        index += len(bt) - 2
                    else:
                        offsets.append((index, index+len(bt)))
                        index += len(bt)
        for i, t in enumerate(bert_question):
            if t != '[UNK]':
                if question[offsets[i][0]:offsets[i][1]] != t.replace("#", "", 2):
                    if question[offsets[i][0]:offsets[i][1]] != '#':
                        is_wrong = True
                        wrong_tokenize += 1
                        break
        if is_wrong:
            continue
        tokenized_question = tokenizer.convert_tokens_to_ids(bert_question)
        tokenized_question.insert(0, 101)
        tokenized_question.append(102)
        
        
        context = dataset['context'][mrc_id]
        mecab_context = mecab.morphs(context)
        offsets = []
        bert_context = []
        index = 0
        for t in mecab_context:
            while context[index] == ' ' or context[index] == '\t' or context[index] == '\n':
                index += 1
            bert_t = tokenizer.tokenize(t)
            if '[UNK]' in bert_t:
                bert_context.append('[UNK]')
                offsets.append((index, index+len(t)))
                index += len(t)
            elif len(bert_t) == 0:
                index += len(t)
            else:
                if index > 0:
                    if context[index-1] != ' ' and context[index-1] != '\t' and context[index-1] != '\n':
                        bert_t[0] = '##' + bert_t[0]
                for bt in bert_t:
                    bert_context.append(bt)
                    if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
                        offsets.append((index, index+len(bt)-2))
                        index += len(bt) - 2
                    else:
                        offsets.append((index, index+len(bt)))
                        index += len(bt)
        for i, t in enumerate(bert_context):
            if t != '[UNK]':
                if context[offsets[i][0]:offsets[i][1]] != t.replace("#", "", 2):
                    if context[offsets[i][0]:offsets[i][1]] != '#':
                        is_wrong = True
                        wrong_tokenize += 1
                        break
        if is_wrong:
            continue
        tokenized_context = tokenizer.convert_tokens_to_ids(bert_context)
                    
        question_length = len(tokenized_question)
        context_length = len(tokenized_context)
        empty_length = input_length - question_length - 1
        index = 0
        while index < context_length:
            input_id = tokenized_question
            token_type_id = [0 for i in range(question_length)]
            if index + empty_length > context_length:
                input_id.extend(tokenized_context[index:context_length])
                offset = offsets[index:context_length]
                index = context_length
            else:
                input_id.extend(tokenized_context[index:index+empty_length])
                offset = offsets[index:index+empty_length]
                index += empty_length - stride
                
            if offset[0][0] > start_char:
                continue
            if offset[-1][1] < end_char:
                continue
                
            input_id.append(102)
            attention_mask = [1 for i in range(len(input_id))]
            for i in range(question_length, len(input_id)):
                token_type_id.append(1)
            for i in range(len(input_id), input_length):
                input_id.append(0)
                token_type_id.append(0)
                attention_mask.append(0)
                    
            token_start_index = 0
            token_end_index = len(offset) - 1
            while offset[token_start_index][0] <= start_char:
                token_start_index += 1
                if token_start_index == len(offset):
                    break
            token_start_index -= 1
            while offset[token_end_index][1] >= end_char:
                token_end_index -= 1
                if token_end_index < 0:
                    break
            token_end_index += 1
                
            if answer["text"][0] != context[offset[token_start_index][0]:offset[token_end_index][1]]:
                print(answer["text"][0], '/', context[offset[token_start_index][0]:offset[token_end_index][1]])
                wrong_answer += 1
            else:
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                start_positions.append(token_start_index)
                end_positions.append(token_start_index)
            
    print('wrong_answer', wrong_answer, 'wrong tokenize', wrong_tokenize)
    return dict(input_ids=input_ids, attention_masks=attention_masks, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)