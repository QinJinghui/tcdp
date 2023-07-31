# -*- encoding: utf-8 -*-
# @Author: Jinghui Qin
# @Time: 2022/12/10
# @File: templates_statistics.py.py

import os
import time
import math
import json
from tqdm import tqdm

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

# data = read_json("Math_23K_Noun_template_no_simplify_N.json")
data = read_json("Math_23K_Noun_template_no_simplify.json")

train_data = read_json("train23k_processed.json")
valid_data = read_json("valid23k_processed.json")
test_data = read_json("test23k_processed.json")

ids = []
template = []
did2tid = {}
for line in tqdm(data):
    if line["template"] not in template:
        template.append(line["template"])
    if "template" in line.keys():
        did2tid[line['id']] = line["template"]
    ids.append(line['id'])

print(len(template), max(template), (0 in template))


train_ids = []
train_template = []
for line in tqdm(train_data):
    train_ids.append(line['id'])
    if line['id'] in did2tid:
        train_template.append(did2tid[line['id']])
print("Train: ", len(set(train_template)))

valid_ids = []
valid_template = []
for line in tqdm(valid_data):
    valid_ids.append(line['id'])
    if line['id'] in did2tid:
        valid_template.append(did2tid[line['id']])
print("Valid: ", len(set(valid_template)))

test_ids = []
test_template = []
for line in tqdm(test_data):
    test_ids.append(line['id'])
    if line['id'] in did2tid:
        test_template.append(did2tid[line['id']])
        if did2tid[line['id']] == 0:
            print("xxxx")
print("Test: ", len(set(test_template)))

print(len(set(train_template)) + len(set(valid_template)) + len(set(test_template)))
print(len(train_template + valid_template + test_template))
print(len(set(train_template + valid_template + test_template)))

print(set(test_template) - set(train_template))