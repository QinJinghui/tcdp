import re
import json
import jieba
import pickle
import jieba.posseg as pseg
from tqdm import tqdm

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f,ensure_ascii=False)

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
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
            # count += 1
            # if count == 100:
                # return data
    return data

def Attr(text, token_list):
    if not isinstance(token_list,list):
        token_list = token_list.strip().split(" ")
    token_attr = [None]*len(token_list)

    words = pseg.cut(text,use_paddle=True) #paddle模式

    words_attr = {}
    for word, flag in words:   
        words_attr[word] = flag

    for i in range(len(token_list)):
        if token_list[i] in words_attr:
            token_attr[i] = words_attr[token_list[i]]

    return token_attr 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
 
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
        seg = d["original_text"].strip().split(" ")
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
            # if "千米/小时" in data_d["equation"]:
            #     data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


if __name__ == "__main__":
    jieba.enable_paddle()
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．"]
    ignore_N = ["体积", "周长", "速度", "距离", "底面积", "面积", "表面积", "鸡", "兔子", "兔", 
        "这", "这个", "这些", "那", "那些", "那个", "底边", "高", "长", "宽", "年利率"]
    separate = [",", "，", ".", "。", "?", "？", "!", "！", "．"]
    data = read_json("hmwp.json")

    pairs, generate_nums, copy_nums = transfer_num(data)
    print(generate_nums)

    train_fold = []
    for p in pairs: # 对表达式进行中序遍历
        train_fold.append((p[0], p[1], p[2], p[3])) #from_infix_to_prefix(p[1])

    formulas = []
    for line in train_fold:
        for i in range(len(line[1])):
            if line[1][i] == '[':
                line[1][i] = '('
            if line[1][i] == ']':
                line[1][i] = ')'
        formulas.append(' '.join(line[1]))
    ###=============== Collecting Template ==============###
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
    save_obj(templates, "template_no_simplify_hmwp")

    ###================denote Noun & Name==================###
    groups = []
    print("processing...")
    max_n = 0
    max_nr = 0

    ig = 0
    n_num = 0
    all_n = 0
    all_word = 0
    index = 1
    for i in tqdm(range(len(data))):
        d = data[i]

        group = {}
        group["old_id"] = d["id"]
        group["id"] = index
        d["old_id"] = d["id"]
        d["id"] = index
        index += 1
        group["noun"] = {}
        group["name"] = {}
        group["group_num"] = []
        group["separate"] = []

        text_token = d["original_text"].strip().split(" ")
        text = ''.join(text_token)
        token_attr = Attr(text, text_token)
        if len(token_attr) != len(text_token) :
            print("CAO NI MA!!!")
        # print(token_attr)
        # print(text_token)
        # n_set = set()
        # nr_set = set()
        tmp_n = 0
        for idx in range(len(text_token)):
            if token_attr[idx] in attr_use  and text_token[idx] not in ignore:
                group["group_num"].append(idx)

            if token_attr[idx] == "n"  and text_token[idx] not in ignore_N:
                tmp_n += 1
                if text_token[idx] not in group["noun"]:      
                    group["noun"][text_token[idx]] = [idx]
                else:
                    group["noun"][text_token[idx]].append(idx)

                    # n_set.add(text_token[idx])
                    # print(text_token[idx])      

            if token_attr[idx] == "nr":
                if text_token[idx] not in group["name"]:      
                    group["name"][text_token[idx]] = [idx]
                else:
                    group["name"][text_token[idx]].append(idx)
                    # nr_set.add(text_token[idx])
                    # print(text_token[idx])
                
            if text_token[idx] in separate:
                group["separate"].append(idx)
        
        if len(group["noun"]) > 5:
            ig += 1
            tmp_n = 0
            group["noun"] = []
        
        if len(group["name"]) <= 1:
            group["name"] = []

        all_n += tmp_n
        all_word += len(text_token)
        
        # group["n_num"] = len(group["noun"])
        # group["nr_num"] = len(group["name"])
        
        group["noun"] = [group["noun"][key] for key in group["noun"]]
        group["name"] = [group["name"][key] for key in group["name"]]

        max_n = max(max_n, len(group["noun"]))
        max_nr = max(max_nr, len(group["name"]))
        groups.append(group)

    print("max N: ", max_n, "  ignore: ", ig, "/", len(data))
    print("maskＮ占比:", all_n/all_word, all_n, all_word)
    print("max Nr: ", max_nr)
    print("saving files...")
    with open("hmwp_Noun.json",'w') as f:
        json.dump(groups,f)

    write_json("hmwp_new.json", data)
    
    ###=============== Generating Template ==============###
    obj = load_obj("template_no_simplify_hmwp")
    values = [d[0] for d in obj]

    obj = read_json("hmwp_Noun.json")
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

    print(obj[0])
    write_json("hmwp_Noun_template_no_simplify.json", obj)
    # {"id": "1", "noun": [], "name": [], "group_num": [3, 5, 8, 13, 19, 21, 22, 39], "separate": [12, 28, 36, 43], "template": 3399}
