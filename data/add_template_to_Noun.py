import json
import jieba
import jieba.posseg as pseg
from tqdm import tqdm

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f)

data_noun = read_json("Math_23K_Noun.json")
data_template =  read_json("Math_23K_attr_template_no_simplify.json")

for i in tqdm(range(len(data_noun))):
    assert data_noun[i]["id"] == data_template[i]["id"]
    data_noun[i]["template"] = data_template[i]["template"]

print(data_noun[100])
write_json("Math_23K_Noun_template_no_simplify.json", data_noun)
