import os
import time
import math
import json
from tqdm import tqdm
import pickle

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

# data = read_json("Math_23K_Noun_template_no_simplify_N.json")
data = read_json("Math_23K_Noun_template_no_simplify.json")

template = []
template_count = {}
template_ids = {}
for line in tqdm(data):
    if line["template"] not in template:
        template.append(line["template"])

    if line["template"] not in template_count:
        template_count[line["template"]] = 0
    template_count[line["template"]] += 1

    if line["template"] not in template_ids:
        template_ids[line["template"]] = []
    template_ids[line["template"]].append(line['id'])

print(len(template), max(template), (0 in template))
print(template_count)
print(sorted(template_count.items(), key = lambda x:x[1], reverse=True))
# print(template_ids)

# def load_template():
#     with open('template.pkl', 'rb') as f:
#         return pickle.load(f)
#
# templates =  load_template()
# print(sorted(templates, key = lambda x:x[1], reverse=True))

# top_template_ids = [1657, 1656, 1655, 1654, 1653, 1652, 1651, 1650, 1649, 1648]
# for t_id in top_template_ids:
#     print(t_id, template_ids[t_id])

top_template_ids = [3576, 3575, 3574, 3573, 3572, 3571, 3570, 3569, 3568, 3567]
for t_id in top_template_ids:
    print(t_id, template_ids[t_id][0])