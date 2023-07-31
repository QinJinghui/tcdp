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

template = []
for line in tqdm(data):
    if line["template"] not in template:
        template.append(line["template"])

print(len(template), max(template), (0 in template))