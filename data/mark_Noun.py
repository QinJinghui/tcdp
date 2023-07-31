import json
import jieba
import jieba.posseg as pseg
from tqdm import tqdm

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

if __name__ == "__main__":
    jieba.enable_paddle()
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．"]
    ignore_N = ["体积", "周长", "速度", "距离", "底面积", "面积", "表面积", "鸡", "兔子", "兔", 
        "这", "这个", "这些", "那", "那些", "那个", "底边", "高", "长", "宽", "年利率"]
    separate = [",", "，", ".", "。", "?", "？", "!", "！", "．"]
    data = load_raw_data("Math_23K.json")
    groups = []
    print("processing...")
    max_n = 0
    max_nr = 0

    ig = 0
    n_num = 0
    all_n = 0
    all_word = 0
    for d in tqdm(data):
        # print(d["original_text"])
        group = {}
        group["id"] = d["id"]
        group["noun"] = {}
        group["name"] = {}
        group["group_num"] = []
        group["separate"] = []
        text = d["original_text"]
        text_token = d["segmented_text"].strip().split(" ")
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
    with open("Math_23K_Noun.json",'w') as f:
        json.dump(groups,f)
