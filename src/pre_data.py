import random
import json
import copy
import re
import nltk
import numpy as np
from copy import deepcopy

from transformers import AutoTokenizer, BertTokenizer, BertConfig, BertModel, AdamW
from torch.utils.data import Dataset

PAD_token = 0


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|\[NUM\]|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words betlow a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["[PAD]", "[NUM]", "[UNK]"] + self.index2word
        else:
            self.index2word = ["[PAD]", "[NUM]"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["[PAD]", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "[UNK]"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["[UNK]"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i



"""
data格式：

{
    "id":"1",
    "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
    "segmented_text":"镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．",
    "equation":"x=(11-1)*2",
    "ans":"20"
}
"""
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
            # if count == 1000:
            #     return data
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


"""
data格式:

{
    "id": "1", 
    "noun": [[3, 5], [8, 19], [21, 22]], 
    "name": [], 
    "group_num": [3, 5, 8, 13, 19, 21, 22, 39], 
    "separate": [12, 28, 36, 43], 
    "n_num": 6, 
    "nr_num": 0
}
"""
def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

# 对表达式进行中序遍历
def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    idx = 0
    for d in data:
        
        # if idx == 100:
        #     break
        # idx += 1

        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def transfer_english_num(data, use_en_roberta=False):  # transfer num into "[NUM]"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        if use_en_roberta:
            # seg = ['<s>'] + nltk.word_tokenize(d["sQuestion"]) + ['</s>']
            seg = nltk.word_tokenize(d["sQuestion"])
        else:
            seg = ['[CLS]'] + nltk.word_tokenize(d["sQuestion"]) + ['[SEP]']
        # seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append(
                {
                    "tokens": input_seq,
                    "output": eq_segs,
                    "nums": nums,
                    "num_idx": num_pos,
                    "id": len(pairs),
                })

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums




# transfer num into "[NUM]"
"""
pairs.append({
        "original_text":d["segmented_text"].strip().split(" "),
        "input":input_seq, # 对数字改为[NUM]后的input列表
        "output":out_seq, # 中序遍历后的表达式
        "nums":nums, # 按顺序记录出现的数字 
        "num_pos":num_pos, # 记录input列表中出现[NUM]的下标
        "id":d["id"],
    })
return pairs, temp_g, copy_nums
"""
def transfer_num(data):  
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


        # 对表达式进行中序遍历
        out_seq = from_infix_to_prefix(out_seq) 
        pairs.append({
            'original_text':d['segmented_text'].strip().split(" "),
            'input':input_seq, # 对数字改为[NUM]后的input列表
            'output':out_seq, # 中序遍历后的表达式
            'nums':nums, # 按顺序记录出现的数字 
            'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
            'id':d['id'],
        })

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

# 合并原本的数据集以及我们自己处理后的数据
#　并同时分出train/val/test
"""
data格式:
{
    'original_text':d['segmented_text'].strip().split(" "),
    'input':input_seq, # 对数字改为[NUM]后的input列表
    'input_MaskN':input_MaskN, # 在input基础上对Noun和Name做Mask后的结果
    'output':out_seq, # 中序遍历后的表达式
    'nums':nums, # 按顺序记录出现的数字 
    'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
    'id':d['id'],
    'noun': g['noun'], # 名词在original_text中的位置
    'name': g['name'], # 名字在original_text中的位置
} 
"""
def get_train_test_fold(ori_path, prefix, raw_data, our_data):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []    
    for pair,g in zip(raw_data, our_data):
        pair['group_num'] = g['group_num']
        pair['separate'] = g['separate']

        pair['noun'] = g['noun']
        pair['name'] = g['name']
        pair['input_MaskN'] = copy.deepcopy(pair['input'])

        pair["template"] = g["template"]

        for i in range(len(pair['noun'])):
            for idx in pair['noun'][i]:
                pair['input_MaskN'][idx] = '[Noun%d]'%(i+1)
        for i in range(len(pair['name'])):
            for idx in pair['name'][i]:
                pair['input_MaskN'][idx] = '[Name%d]'%(i+1)

        # pair.append(g['template'])
        if pair['id'] in train_id:
            train_fold.append(pair)
        elif pair['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold
    

"""
data格式:{
    'original_text':d['segmented_text'].strip().split(" "),
    'input':input_seq, # 对数字改为[NUM]后的input列表
    'input_MaskN':input_MaskN, # 在input基础上对Noun和Name做Mask后的结果
    'output':out_seq, # 中序遍历后的表达式
    'nums':nums, # 按顺序记录出现的数字 
    'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
    'id':d['id'],
    'noun': g['noun'], # 名词在original_text中的位置
    'name': g['name'], # 名字在original_text中的位置
} 
"""
def process_data_pipeline(raw_data_path, our_data_path, ori_path, prefix, debug=False):
    raw_data = load_raw_data(raw_data_path)
    """
    raw_data：
    {
        "id":"1",
        "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
        "segmented_text":"镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．",
        "equation":"x=(11-1)*2",
        "ans":"20"
    }
    """
    our_data = read_json(our_data_path)
    """
    our_data:
    {
        "id": "1", 
        "noun": [[3, 5], [8, 19], [21, 22]], 
        "name": [], 
        "group_num": [3, 5, 8, 13, 19, 21, 22, 39], 
        "separate": [12, 28, 36, 43], 
        "n_num": 6, 
        "nr_num": 0,
        "template": 0-2000
    }
    """
    
    ##########################################
    if debug:
        raw_data = raw_data[:1000]
        our_data = our_data[:1000]
    ##########################################
    
    raw_data, generate_nums, copy_nums = transfer_num(raw_data)
    """
    pairs.append({
        "original_text":d["segmented_text"].strip().split(" "),
        "input":input_seq, # 对数字改为[NUM]后的input列表
        "output":out_seq, # 中序遍历后的表达式
        "nums":nums, # 按顺序记录出现的数字 
        "num_pos":num_pos, # 记录input列表中出现[NUM]的下标
        "id":d["id"],
    })
    return pairs, temp_g, copy_nums
    """
    train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, raw_data, our_data)
    """
    data格式:{
        'original_text':d['segmented_text'].strip().split(" "),
        'input':input_seq, # 对数字改为[NUM]后的input列表
        'input_MaskN':input_MaskN, # 在input基础上对Noun和Name做Mask后的结果
        'output':out_seq, # 中序遍历后的表达式
        'nums':nums, # 按顺序记录出现的数字 
        'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
        'id':d['id'],
        'noun': g['noun'], # 名词在original_text中的位置
        'name': g['name'], # 名字在original_text中的位置
        "template": 模板编号
    } 
    """
    return train_fold, test_fold, valid_fold, generate_nums, copy_nums


"""
data格式:{
    "tokens":tokens,
    "tokens_MaskN":tokens_MaskN,
    "num_idx":num_idx,
    "noun":[noun[key] for key in noun],
    "name":[name[key] for key in name],
}
"""
def indexes_from_sentence_input(lang, pair, tokenizer, tree=False):
    # [CLS] + tokens + [SEP]

    pair['input'] = ['[CLS]'] + pair['input'] + ['[SEP]']
  
    noun = {} # {'[Noun1]':[0], '[Noun2]':[1], '[Noun3]':[2], '[Noun4]':[3], '[Noun5]':[4]}
    name = {}
    
    """
    data格式:{
        'original_text':d['segmented_text'].strip().split(" "),
        'input':input_seq, # 对数字改为[NUM]后的input列表
        'input_MaskN':input_MaskN, # 在input基础上对Noun和Name做Mask后的结果
        'output':out_seq, # 中序遍历后的表达式
        'nums':nums, # 按顺序记录出现的数字 
        'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
        'id':d['id'],
        'noun': g['noun'], # 名词在original_text中的位置
        'name': g['name'], # 名字在original_text中的位置
        "template": 模板编号
    } 
    """
    idx = 0
    num_idx = []
    tokens = [] 
    tokens_MaskN = []
    for word, wordM in zip(pair['input'], pair['input_MaskN']):
        if len(word) == 0: #　跳过空的 
            continue

        elif word == '[NUM]':
            num_idx.append(idx)
            tokens.append(word)
            tokens_MaskN.append(word)
            idx += 1

        elif wordM in ['[Noun1]', '[Noun2]', '[Noun3]', '[Noun4]', '[Noun3]']:
            tokens.append(word)
            tokens_MaskN.append('[NounX]')
            if wordM not in noun:
                noun[wordM] = [idx]
            else:
                noun[wordM].append(idx)
            idx += 1

        elif wordM in ['[Name1]', '[Name2]']:
            tokens.append(word)
            tokens_MaskN.append('[NameX]')
            if wordM not in name:
                name[wordM] = [idx]
            else:
                name[wordM].append(idx)
            idx += 1

        else:
            tmp_token = tokenizer.tokenize(word) #[UNK]
            if tmp_token == []:
                print("processing ", word, [word], pair['id'])
                tmp_token = ['[UNK]']
            tokens += tmp_token
            tokens_MaskN += tmp_token
            # tmp = np.linspace(idx, idx+len(tmp_token)-1, num=len(tmp_token)).astype(np.int16).tolist()
            # assert tmp != []
            idx += len(tmp_token)
            
    line = {
        "tokens":tokens,
        "tokens_MaskN":tokens_MaskN,
        "num_idx":num_idx,
        "noun":[noun[key] for key in noun],
        "name":[name[key] for key in name],
    }
    return line



def indexes_from_sentence_output(lang, sentence, tree=False):
    res = []
    idx = 0
    for word in sentence:
        if len(word) == 0:
            print(sentence)
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
            idx += 1
        else:
            res.append(lang.word2index["[UNK]"])
            idx += 1
        
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


"""
data格式:{
    "tokens":tokens,
    "tokens_MaskN":tokens_MaskN,
    "num_idx":num_idx,
    "noun":[noun[key] for key in noun], 
    "name":[name[key] for key in name],
    "token_ids": ,
    "token_type_ids": ,
    "attention_mask": ,
    "output": ,
    "num_stack": ,
    "nums": ,
    "id": ,
    "original_text": ,
    "template": ,
}
"""
def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, 
                      copy_nums, path, max_seq_length, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    # Load Bert-chinese-wwm tokenizer 
    tokenizer = BertTokenizer.from_pretrained(path)
    PAD_id = tokenizer.pad_token_id

    ## build lang
    print("Tokenizing/Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair['input'])
        output_lang.add_sen_to_vocab(pair['output'])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    
    """
    data格式:{
        'original_text':d['segmented_text'].strip().split(" "),
        'input':input_seq, # 对数字改为[NUM]后的input列表
        'input_MaskN':input_MaskN, # 在input基础上对Noun和Name做Mask后的结果
        'output':out_seq, # 中序遍历后的表达式
        'nums':nums, # 按顺序记录出现的数字 
        'num_pos':num_pos, # 记录input列表中出现[NUM]的下标
        'id':d['id'],
        'noun': g['noun'], # 名词在original_text中的位置
        'name': g['name'], # 名字在original_text中的位置
        "template": 模板编号
    } 
    """
    ## 逐行处理
    for pair in pairs_trained:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                # print("output lang not find: ", word)
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        """
        line:{
            "tokens":tokens,
            "tokens_MaskN":tokens_MaskN,
            "num_idx":num_idx,
            "noun":[noun[key] for key in noun],
            "name":[name[key] for key in name],
        }
        """
        ##　此处的numpos, group, ..., noun等都是基于input_cell的下标
        line = indexes_from_sentence_input(input_lang, pair, tokenizer, tree=tree) 
        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree)   

        token_ids = tokenizer.convert_tokens_to_ids(line["tokens"])
        token_len = len(token_ids)
        line['token_len'] = token_len
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids
        
        ### Testing num 
        for idx in line["num_idx"]:
            assert line["tokens"][idx] == '[NUM]'

        num_id = [tokenizer.convert_tokens_to_ids(['[NUM]'])[0]]
        num_pos = []
        for idx, t_id in enumerate(token_ids):
            if t_id in num_id:
                num_pos.append(idx)

        pair["num_idx"] = num_pos # update num pos after plm tokenizer

        for row in line["noun"]:
            for idx in row:
                assert line["tokens_MaskN"][idx] == '[NounX]'
        for row in line["name"]:
            for idx in row:
                assert line["tokens_MaskN"][idx] == '[NameX]'
        
        line["token_ids"] = token_ids
        line["token_type_ids"] = token_type_ids
        line["attention_mask"] = attention_mask
        line["output"] = output_cell
        line["num_stack"] = num_stack
        line["nums"] = pair["nums"]
        line["id"] = pair["id"]
        line["original_text"] = pair["original_text"]
        line["template"] = pair["template"]

        train_pairs.append(line)

        """
        line{
            "tokens":tokens,
            "tokens_MaskN":tokens_MaskN,
            "num_idx":num_idx,
            "noun":[noun[key] for key in noun],
            "name":[name[key] for key in name],
            "token_ids": ,
            "token_type_ids": ,
            "attention_mask": ,
            "output": ,
            "num_stack": ,
            "nums": ,
            "id": ,
            "original_text": ,
            "template": ,
        }
        """
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    ## 逐行处理
    for pair in pairs_tested:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                # print("output lang not find: ", word)
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])
        num_stack.reverse()
        
        ##　此处的numpos, group, ..., noun等都是基于input_cell的下标
        line = indexes_from_sentence_input(input_lang, pair, tokenizer, tree=tree) 
        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree)

        token_ids = tokenizer.convert_tokens_to_ids(line["tokens"])
        token_len = len(token_ids)
        line['token_len'] = token_len
        # Padding 
        padding_ids = [PAD_id] * (max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0] * max_seq_length
        # attention_mask
        attention_mask = [1] * token_len + padding_ids
        
        ### Testing num 
        for idx in line["num_idx"]:
            assert line["tokens"][idx] == '[NUM]'

        num_id = [tokenizer.convert_tokens_to_ids(['[NUM]'])[0]]
        num_pos = []
        for idx, t_id in enumerate(token_ids):
            if t_id in num_id:
                num_pos.append(idx)

        pair["num_idx"] = num_pos # update num pos after plm tokenizer

        for row in line["noun"]:
            for idx in row:
                assert line["tokens_MaskN"][idx] == '[NounX]'
        for row in line["name"]:
            for idx in row:
                assert line["tokens_MaskN"][idx] == '[NameX]'
        
        line["token_ids"] = token_ids
        line["token_type_ids"] = token_type_ids
        line["attention_mask"] = attention_mask
        line["output"] = output_cell
        line["num_stack"] = num_stack
        line["nums"] = pair["nums"]
        line["id"] = pair["id"]
        line["original_text"] = pair["original_text"]
        line["template"] = pair["template"]

        test_pairs.append(line)

    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs


def prepare_data_Batch(data, batch_size):
    Noun = [2, 3, 4, 5, 6]
    Name = [7, 8]
    
    data = copy.deepcopy(data)
    ## 准备数据batch
    random.shuffle(data)

    pos = 0
    batches = []
    while pos + batch_size < len(data):
        batches.append(data[pos:pos+batch_size])
        pos += batch_size
    batches.append(data[pos:])

    data = []
    for batch in batches:
        token_len = []
        token_ids = []
        token_type_ids = []
        attention_mask = []
        output = []
        output_len = []
        nums = []
        num_size = []
        num_idx = []
        num_stack = []
        token_ids_MaskN = []
        token_ids_DisturbN = []
        template = []

        for line in batch:
            random.shuffle(Noun)
            random.shuffle(Name)

            # token_len.append(len(line["tokens"]))
            token_len.append(line["token_len"])
            token_ids.append(line["token_ids"])
            token_type_ids.append(line["token_type_ids"])
            attention_mask.append(line["attention_mask"])
            output.append(line["output"])
            output_len.append(len(line["output"]))
            template.append(line["template"])
            nums.append(line["nums"])
            num_size.append(len(line["nums"]))
            num_idx.append(line["num_idx"])
            num_stack.append(line["num_stack"])

            tmp_ids = copy.deepcopy(line["token_ids"])
            for i in range(len(line["noun"])):
                for j in line["noun"][i]:
                    tmp_ids[j] = Noun[i]
            for i in range(len(line["name"])):
                for j in line["name"][i]:
                    tmp_ids[j] = Name[i]
            token_ids_MaskN.append(tmp_ids)

            tmp_ids2 = copy.deepcopy(line["token_ids"])
            for i in range(len(line["noun"])):
                for j in line["noun"][i]:
                    tmp_ids2[j] = Noun[random.randint(0, 4)]
            for i in range(len(line["name"])):
                for j in line["name"][i]:
                    tmp_ids2[j] = Name[random.randint(0, 1)]
            token_ids_DisturbN.append(tmp_ids2)

        
        data_batch = {
            "max_token_len": max(token_len),
            "token_ids": [line[:max(token_len)] for line in token_ids],
            "token_type_ids": [line[:max(token_len)] for line in token_type_ids],
            "attention_mask": [line[:max(token_len)] for line in attention_mask],
            "output": pad_seq(output, max(output_len)),
            "output_len":output_len,
            "nums": nums,
            "num_size":num_size,
            "num_idx": num_idx,
            "num_stack": num_stack,
            "token_ids_MaskN": [line[:max(token_len)] for line in token_ids_MaskN],
            "token_ids_DisturbN": [line[:max(token_len)] for line in token_ids_DisturbN],
            "template": template,
        }
    
        data.append(data_batch)
    return data

def pad_seq(seq, max_length):
    PAD_token = 0
    seq = [line+[PAD_token for _ in range(max_length-len(line))] for line in seq]
    # seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq

def generate_template_hmwp(pairs):
    formulas = []
    for line in pairs:
        for i in range(len(line["output"])):
            if line["output"][i] == '[':
                line["output"][i] = '('
            if line["output"][i] == ']':
                line["output"][i] = ')'
        formulas.append(' '.join(line["output"]))

    templates = {}
    for i in range(len(formulas)):
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

    # print("模板数量: ", len(values))     
    # print("示例： ", templates[100])     
    # print("示例： ", templates[200])     
    # print("================================================")     

    for i in range(len(pairs)):
        line = copy.deepcopy(pairs[i])
        for j in range(len(line["output"])):
            if line["output"][j] == '[':
                line["output"][j] = '('
            if line["output"][j] == ']':
                line["output"][j] = ')'

        pairs[i]["template"] = formulas.index(' '.join(line["output"]))
        pairs[i]["output"] = from_infix_to_prefix(pairs[i]["output"])

    return pairs

def pos_tagging(pairs):
    Punct = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．" ,'[CLS]', '[SEP]', '[NUM]']
    Noun_ignore = ["体积", "周长", "速度", "距离", "底面积", "面积", "表面积", "鸡", "兔子", "兔", 
        "这", "这个", "这些", "那", "那些", "那个", "底边", "高", "长", "宽", "年利率", '[CLS]', '[SEP]', '[NUM]']
    
    POS_N = ["NN", "NNS", "NNP", "NNPS"]

    for i in range(len(pairs)):
        tokens = pairs[i]["tokens"]
        pos_ = nltk.pos_tag(tokens)
        ne = nltk.ne_chunk(pos_)
        pos = [item[1] for item in pos_]
        assert len(pos) == len(tokens)
        # if len(ne) != len(pos):
        #     print(tokens)
        #     print(pos_)
        #     print(ne)
        #     print(len(pos), len(list(ne)))
        #     print(ne[2], len(ne[2]))
        #     break
         
        noun = {} # 记录noun/name的token 
        name = {}

        idx = 0
        for item in ne:
            if hasattr(item, '_label') and (item._label == "PERSON"):
                for item_t in item:
                    assert tokens[idx] == item_t[0]

                    if item_t[0] not in Punct+Noun_ignore:
                        if item_t[0] not in name:
                            name[item_t[0]] = [idx]
                        else:
                            name[item_t[0]].append(idx)
                    idx += 1

            elif hasattr(item, '_label'):
                idx += len(item)
            else:
                idx += 1
        
        for idx in range(len(pos)):
            if (pos[idx] in POS_N) and (tokens[idx] not in name) and (tokens[idx] not in Punct+Noun_ignore):
                if tokens[idx] not in noun:
                    noun[tokens[idx]] = [idx]
                else:    
                    noun[tokens[idx]].append(idx)

        pairs[i]["noun"] = [noun[key] for key in noun]
        pairs[i]["name"] = [name[key] for key in name]

        pairs[i]["tokens_MaskN"] = copy.deepcopy(pairs[i]["tokens"])
        for noun_line in pairs[i]["noun"]:
            for idx in noun_line:
                pairs[i]["tokens_MaskN"][idx] = '[NounX]'
        for name_line in pairs[i]["name"]:
            for idx in name_line:
                pairs[i]["tokens_MaskN"][idx] = '[NameX]'

        if len(pairs[i]["noun"]) > 5:
            pairs[i]["noun"] = []
        if len(pairs[i]["name"]) >2:
            pairs[i]["name"] = []

    return pairs

"""
{
    'tokens': ['[CLS]', 'Bryan', 'took', 'have', 'in', 'total', '?', '[SEP]'], 
    "tokens_MaskN": [],
    'output': ['*', 'N0', 'N1'], 
    'nums': ['56', '9'], 
    'num_idx': [14, 20], 
    'id': 0, 
    'template': 0, 
    'noun': [[4], [7, 15, 25], [21]], 
    'name': [[1, 12]]
}
"""
def process_data_pipeline_hmwp(data_path, use_en_roberta=False):
    data = load_mawps_data(data_path)
    pairs, generate_nums, copy_nums = transfer_english_num(data, use_en_roberta=use_en_roberta)
    pairs = generate_template_hmwp(pairs)
    pairs = pos_tagging(pairs)

    return pairs, generate_nums, copy_nums


    
"""
data格式:{
    'tokens', 
    'output', 
    'nums', 
    'num_idx', 
    'id', 
    'template', 
    'noun', 
    'name', 
    'tokens_MaskN', 
    'token_ids', 
    'token_type_ids', 
    'attention_mask', 
    'num_stack'
}
"""
def prepare_data_hmwp(pairs_trained, pairs_tested, generate_nums, 
                      copy_nums, tokenizer, max_seq_length, tree=False, use_en_roberta=False): # path
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    # if use_en_roberta:
    #     max_seq_length += 2
    # Load Bert-chinese-wwm tokenizer 
    # tokenizer = BertTokenizer.from_pretrained(path)
    # tokenizer = AutoTokenizer.from_pretrained(path)
    PAD_id = tokenizer.pad_token_id

    ## build lang
    print("Tokenizing/Indexing words...")
    for pair in pairs_trained:
        output_lang.add_sen_to_vocab(pair['output'])
    
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums) 
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    
    """
    data格式:{
        'tokens': ['[CLS]', 'Bryan', 'took', 'have', 'in', 'total', '?', '[SEP]'], 
        'output': ['*', 'N0', 'N1'], 
        'nums': ['56', '9'], 
        'num_idx': [14, 20], 
        'id': 0, 
        'template': 0, 
        'noun': [[4], [7, 15, 25], [21]], 
        'name': [[1, 12]]
    }
    """
    ## 逐行处理
    for pair in pairs_trained:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                # print("output lang not find: ", word)
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()

        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree)

        if use_en_roberta:
            tokens = ' '.join(pair["tokens"]).replace('[NUM]', 'NUM').replace('’', '\'').split()
            # token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_dict = tokenizer(' '.join(tokens))
            token_ids = token_dict['input_ids']
            # print(tokenizer.convert_ids_to_tokens(token_ids))
            # exit(0)
            # print(tokenizer.convert_ids_to_tokens(token_ids['input_ids']))
        else:
            token_ids = tokenizer.convert_tokens_to_ids(pair["tokens"])
        token_len = len(token_ids)
        pair['token_len'] = token_len
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids
        
        ### Testing num
        for idx in pair["num_idx"]:
            assert pair["tokens"][idx] == '[NUM]'

        if use_en_roberta:
            num_id = [tokenizer.convert_tokens_to_ids(['ĠNUM'])[0], tokenizer.convert_tokens_to_ids(['NUM'])[0]]
        else:
            num_id = [tokenizer.convert_tokens_to_ids(['[NUM]'])[0]]
        num_pos = []
        for idx, t_id in enumerate(token_ids):
            if t_id in num_id:
                num_pos.append(idx)

        if len(pair["num_idx"]) != len(num_pos):
            print(token_ids)
            print(tokenizer.convert_ids_to_tokens(token_ids))
            print(num_id)
            print(pair["num_idx"])
            print(num_pos)
            exit(0)

        pair["num_idx"] = num_pos # update num pos after plm tokenizer

        for row in pair["noun"]:
            for idx in row:
                assert pair["tokens_MaskN"][idx] == '[NounX]'
        for row in pair["name"]:
            for idx in row:
                assert pair["tokens_MaskN"][idx] == '[NameX]'
        
        pair["token_ids"] = token_ids
        pair["token_type_ids"] = token_type_ids
        pair["attention_mask"] = attention_mask
        pair["output"] = output_cell
        pair["num_stack"] = num_stack
        train_pairs.append(pair)

    print('Indexed %d words in output' % (output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

        ## 逐行处理
    for pair in pairs_tested:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                # print("output lang not find: ", word)
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()

        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree)   

        # token_ids = tokenizer.convert_tokens_to_ids(pair["tokens"])
        if use_en_roberta:
            # tokens = pair["tokens"].replace('[NUM]', 'NUM').replace('’', '\'')
            # token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tokens = ' '.join(pair["tokens"]).replace('[NUM]', 'NUM').replace('’', '\'').split()
            # token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_dict = tokenizer(' '.join(tokens))
            token_ids = token_dict['input_ids']
            # print(tokenizer.convert_ids_to_tokens(token_ids))
            # print(tokenizer.convert_ids_to_tokens(temp_dict['input_ids']))
        else:
            token_ids = tokenizer.convert_tokens_to_ids(pair["tokens"])
        token_len = len(token_ids)
        pair['token_len'] = token_len
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids
        
        ### Testing num 
        for idx in pair["num_idx"]:
            assert pair["tokens"][idx] == '[NUM]'

        if use_en_roberta:
            num_id = [tokenizer.convert_tokens_to_ids(['ĠNUM'])[0], tokenizer.convert_tokens_to_ids(['NUM'])[0]]
        else:
            num_id = [tokenizer.convert_tokens_to_ids(['[NUM]'])[0]]
        num_pos = []
        for idx, t_id in enumerate(token_ids):
            if t_id in num_id:
                num_pos.append(idx)

        if len(pair["num_idx"]) != len(num_pos):
            print(token_ids)
            print(tokenizer.convert_ids_to_tokens(token_ids))
            print(num_id)
            print(pair["num_idx"])
            print(num_pos)
            exit(0)

        pair["num_idx"] = num_pos # update num pos after plm tokenizer

        for row in pair["noun"]:
            for idx in row:
                assert pair["tokens_MaskN"][idx] == '[NounX]'
        for row in pair["name"]:
            for idx in row:
                assert pair["tokens_MaskN"][idx] == '[NameX]'
        
        pair["token_ids"] = token_ids
        pair["token_type_ids"] = token_type_ids
        pair["attention_mask"] = attention_mask
        pair["output"] = output_cell
        pair["num_stack"] = num_stack

        test_pairs.append(pair)
    
    print('Number of testing data %d' % (len(test_pairs)))

    return output_lang, train_pairs, test_pairs