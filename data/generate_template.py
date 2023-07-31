# # coding: utf-8
# import sys 
# sys.path.append("..") 
# from src.train_and_evaluate import *
# from src.pre_data import *
import os
import time
import math
from tqdm import tqdm
# from src.expressions_transfer import *
import json
import sys
sys.path.append("..")
# from src.log_utils import *
import random
import json
import copy
import re
import numpy as np

max_seq_length = 180
batch_size = 8
embedding_size = 128
hidden_size = 768
n_epochs = 100
learning_rate_bert = 5e-5 #3e-5 # 2e-5
weight_decay_bert = 1e-5 #2e-5 #2e-5 # 1e-5
warmup_proportion = 0.1 
learning_rate = 5e-5
weight_decay = 1e-5
beam_size = 1
n_layers = 2
max_animate_times = 2
no_animate_epochs = 0
ori_path = './data/'
prefix = '23k_processed.json'

Exist_template = True

# log_file = os.path.join(save_path, 'log')
# create_logs(log_file)

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
 
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f)

def transfer_num(data):  # transfer num into "[NUM]"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
            
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    count = 0
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            count += 1
            # if count == 100:
            #     return data
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def get_train_test_fold(prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = mode_train + prefix
    valid_path = mode_valid + prefix
    test_path = mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        # (input_seq, out_seq, nums, num_pos, group_num, separate)
        pair.append(g['group_num'])
        pair.append(g['separate'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

# def combine_json(graph_dict, ori_json):
#     new_data = []
#     for i in tqdm(range(len(ori_json))):
#         item = ori_json[i]
#         item['group_num'] = graph_dict[item['id']]['group_num']
#         new_data.append(item)        
#     print('Graph has been inserted into ori_json!')
#     return new_data


data = load_raw_data("Math_23K.json")
group_data = read_json("Math_23K_attr.json")

pairs, generate_nums, copy_nums = transfer_num(data)
print(generate_nums)

temp_pairs = []
for p in pairs: # 对表达式进行中序遍历
    temp_pairs.append((p[0], p[1], p[2], p[3])) #from_infix_to_prefix(p[1])
pairs = temp_pairs

# 根据train/text/valid文件索引id，打包pair和group_num
# [WordToken, Expression, NUM_str, NUM_pos, GroupIdx]
train_fold, test_fold, valid_fold = get_train_test_fold(prefix,data,pairs,group_data)
# print(train_fold[0])
# print(train_fold[0][1], len(train_fold))

formulas = []
for line in train_fold:
    for i in range(len(line[1])):
        if line[1][i] == '[':
            line[1][i] = '('
        if line[1][i] == ']':
            line[1][i] = ')'
    formulas.append(' '.join(line[1]))


###=============== Collecting Template ==============###
if not Exist_template:
    templates = {}

    from sympy import sympify, simplify
    for i in tqdm(range(len(formulas))):
        ##===== simplify =====##
        """
        line = sympify(formulas[i], evaluate=False)
        formulas[i] = simplify(line, evaluate=False).__str__()
        if formulas[i] not in templates:
            templates[formulas[i]] = 1
        else:
            templates[formulas[i]] += 1
        """
        ##===== no simplify =====##
        if formulas[i] not in templates:
            templates[formulas[i]] = 1
        else:
            templates[formulas[i]] += 1

    values = sorted(templates.values())
    templates = sorted(templates.items(), key=lambda d: d[1], reverse=False)

    print("模板数量: ", len(values)) 
    print("示例： ", templates[100])
    print("示例： ", templates[200])
    save_obj(templates, "template_no_simplify")


###=============== Generating Template ==============###
formulas = []
for line in pairs:
    for i in range(len(line[1])):
        if line[1][i] == '[':
            line[1][i] = '('
        if line[1][i] == ']':
            line[1][i] = ')'
    formulas.append(' '.join(line[1]))

obj = load_obj("template_no_simplify")
values = [d[0] for d in obj]
    
obj = read_json("Math_23K_attr.json")
from sympy import sympify, simplify
for i in tqdm(range(len(formulas))):
    ##===== simplify =====##
    """
    line = sympify(formulas[i], evaluate=False)
    formulas[i] = simplify(line, evaluate=False).__str__()
    if formulas[i] in values:
        obj[i]["template"] = values.index(formulas[i])+1
    else:
        obj[i]["template"] = 0
    """

    ##===== no simplify =====##
    if formulas[i] in values:
        obj[i]["template"] = values.index(formulas[i])+1
    else:
        obj[i]["template"] = 0
write_json("Math_23K_attr_template_no_simplify.json", obj)
