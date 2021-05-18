#######################################################################################################################
# prepare_train_features -> accurate_prepare_train_features로 대체해서 사용
# prepare_validation_features -> accurate_prepare_validation_features로 대체해서 사용
# *주의* map 옵션에 num_procs = 12 사용하면 안 됨
# klue와 korquad 합친 dataset을 tokenize할 때 대략 25분 소요되기 때문에
# 생성한 train_dataset을 bin 파일로 저장하고 load 해서 사용하는 것을 추천합니다.
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
    wrong_context = 0
    wrong_question = 0
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
                if t[0] == '\u200e' or t[0] == '\xad':
                    index += 1
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
                        wrong_question += 1
                        print(offsets[i], question[offsets[i][0]:offsets[i][1]], t.replace("#", "", 2))
                        break
        if is_wrong:
            print(dataset['question'][mrc_id])
            print(mecab_question)
            for t, o in zip(bert_question, offsets):
                print(f'{t}:{o} ', end=' ')
            continue                
        tokenized_question = tokenizer.convert_tokens_to_ids(bert_question)
        tokenized_question.insert(0, 101)
        tokenized_question.append(102)
    
        is_wrong = False
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
                if t[0] == '\u200e' or t[0] == '\xad':
                    index += 1
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
                        wrong_context += 1
                        print(offsets[i], context[offsets[i][0]:offsets[i][1]], t.replace("#", "", 2))
                        break
        if is_wrong:
            print(dataset['context'][mrc_id])
            print(mecab_context)
            for t, o in zip(bert_context, offsets):
                print(f'{t}:{o} ', end=' ')
            continue
        tokenized_context = tokenizer.convert_tokens_to_ids(bert_context)
                    
        question_length = len(tokenized_question)
        context_length = len(tokenized_context)
        empty_length = input_length - question_length - 1
        index = 0
        while index < context_length:
            input_id = copy.deepcopy(tokenized_question)
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
                break
            else:
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                start_positions.append(token_start_index + question_length)
                end_positions.append(token_end_index + question_length)
            
    print('wrong_question', wrong_question, 'wrong_answer', wrong_answer, 'wrong_context', wrong_context)
    return dict(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)

def accurate_prepare_validation_features(dataset):
    mecab = Mecab()
    input_ids = []
    attention_masks = []
    token_type_ids = []
    offset_mappings = []
    example_ids = []
    input_length = 512
    wrong_context = 0
    wrong_question = 0
    stride = 128
    
    for mrc_id in range(len(dataset['context'])):
        is_wrong = False
            
        question = dataset['question'][mrc_id]
        mecab_question = mecab.morphs(question)
        offset_mapping = []
        bert_question = []
        index = 0
        for t in mecab_question:
            while question[index] == ' ' or question[index] == '\t' or question[index] == '\n':
                index += 1
            bert_t = tokenizer.tokenize(t)
            if '[UNK]' in bert_t:
                bert_question.append('[UNK]')
                offset_mapping.append((index, index+len(t)))
                index += len(t)
            elif len(bert_t) == 0:
                index += len(t)
            else:
                if index > 0:
                    if question[index-1] != ' ' and question[index-1] != '\t' and question[index-1] != '\n':
                        bert_t[0] = '##' + bert_t[0]
                if t[0] == '\u200e' or t[0] == '\xad':
                    index += 1
                for bt in bert_t:
                    bert_question.append(bt)
                    if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
                        offset_mapping.append((index, index+len(bt)-2))
                        index += len(bt) - 2
                    else:
                        offset_mapping.append((index, index+len(bt)))
                        index += len(bt)
        for i, t in enumerate(bert_question):
            if t != '[UNK]':
                if question[offset_mapping[i][0]:offset_mapping[i][1]] != t.replace("#", "", 2):
                    if question[offset_mapping[i][0]:offset_mapping[i][1]] != '#':
                        is_wrong = True
                        wrong_question += 1
                        print(offset_mapping[i], question[offset_mapping[i][0]:offset_mapping[i][1]], t.replace("#", "", 2))
                        break
        if is_wrong:
            print(dataset['question'][mrc_id])
            print(mecab_question)
            for t, o in zip(bert_question, offset_mapping):
                print(f'{t}:{o} ', end=' ')
        tokenized_question = tokenizer.convert_tokens_to_ids(bert_question)
        tokenized_question.insert(0, 101)
        tokenized_question.append(102)
    
        is_wrong = False
        context = dataset['context'][mrc_id]
        mecab_context = mecab.morphs(context)
        offset_mapping = []
        bert_context = []
        index = 0
        for t in mecab_context:
            while context[index] == ' ' or context[index] == '\t' or context[index] == '\n':
                index += 1
            bert_t = tokenizer.tokenize(t)
            if '[UNK]' in bert_t:
                bert_context.append('[UNK]')
                offset_mapping.append((index, index+len(t)))
                index += len(t)
            elif len(bert_t) == 0:
                index += len(t)
            else:
                if index > 0:
                    if context[index-1] != ' ' and context[index-1] != '\t' and context[index-1] != '\n':
                        bert_t[0] = '##' + bert_t[0]
                if t[0] == '\u200e' or t[0] == '\xad':
                    index += 1
                for bt in bert_t:
                    bert_context.append(bt)
                    if len(bt) >= 3 and bt[0] == '#' and bt[1] == '#':
                        offset_mapping.append((index, index+len(bt)-2))
                        index += len(bt) - 2
                    else:
                        offset_mapping.append((index, index+len(bt)))
                        index += len(bt)
        for i, t in enumerate(bert_context):
            if t != '[UNK]':
                if context[offset_mapping[i][0]:offset_mapping[i][1]] != t.replace("#", "", 2):
                    if context[offset_mapping[i][0]:offset_mapping[i][1]] != '#':
                        is_wrong = True
                        wrong_context += 1
                        print(offset_mapping[i], context[offset_mapping[i][0]:offset_mapping[i][1]], t.replace("#", "", 2))
                        break
        if is_wrong:
            print(dataset['context'][mrc_id])
            print(mecab_context)
            for t, o in zip(bert_context, offset_mapping):
                print(f'{t}:{o} ', end=' ')
        tokenized_context = tokenizer.convert_tokens_to_ids(bert_context)
                    
        question_length = len(tokenized_question)
        context_length = len(tokenized_context)
        empty_length = input_length - question_length - 1
        index = 0
        while index < context_length:
            input_id = copy.deepcopy(tokenized_question)
            token_type_id = [0 for i in range(question_length)]
            example_offset_mapping = [None for i in range(question_length)]
            if index + empty_length > context_length:
                input_id.extend(tokenized_context[index:context_length])
                example_offset_mapping.extend(offset_mapping[index:context_length])
                index = context_length
            else:
                input_id.extend(tokenized_context[index:index+empty_length])
                example_offset_mapping.extend(offset_mapping[index:index+empty_length])
                index += empty_length - stride
                
            input_id.append(102)
            attention_mask = [1 for i in range(len(input_id))]
            for i in range(question_length, len(input_id)):
                token_type_id.append(1)
            for i in range(len(input_id), input_length):
                input_id.append(0)
                token_type_id.append(0)
                attention_mask.append(0)
                example_offset_mapping.append(None)
                    
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            offset_mappings.append(example_offset_mapping)
            example_ids.append(dataset["id"][mrc_id])
            
    print('wrong_question', wrong_question, 'wrong_context', wrong_context)
    return dict(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, offset_mapping=offset_mappings, example_id=example_ids)