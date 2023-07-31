import json

# coding: utf-8
import sys
sys.path.append("..")
from src.train_and_evaluate import *
from src.models import *
import os
import time
import math
import torch.optim
from src.expressions_transfer import *
import json
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file
# tmp = {
#     "id":"1",
#     "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
#     "segmented_text":"镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．",
#     "equation":"x=(11-1)*2",
#     "ans":"20"
# }
# tmp = [tmp,tmp]
# json_str = json.dumps(tmp, indent=4, ensure_ascii=False)
# with open('test_data.json', 'w') as json_file:
#     json_file.write(json_str)

data = load_raw_data("Math_23K.json")
group_data = read_json("Math_23K_attr.json")

pairs, generate_nums, copy_nums = transfer_num(data)

print(pairs)