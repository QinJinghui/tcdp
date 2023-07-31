import json
import jieba
import jieba.posseg as pseg


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
    separate = [",", "，", ".", "。", "?", "？", "!", "！", "．"]
    data = load_raw_data("Math_23K.json")
    groups = []
    print("processing...")
    for d in data:
        # print(d["original_text"])
        group = {}
        group["id"] = d["id"]
        group["group_num"] = []
        group["separate"] = []
        text = d["original_text"]
        text_token = d["segmented_text"].strip().split(" ")
        token_attr = Attr(text, text_token)
        for idx in range(len(text_token)):
            if token_attr[idx] in attr_use  and  not text_token[idx] in ignore:
                group["group_num"].append(idx)
                # print(text_token[idx], token_attr[idx])
            if text_token[idx] in separate:
                group["separate"].append(idx)
        groups.append(group)
    
    print("saving files...")
    with open("Math_23K_attr.json",'w') as f:
        json.dump(groups,f)
