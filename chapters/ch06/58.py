import os
from dotenv import load_dotenv
import gensim
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../data/ch06/questions-words.txt"
data_path = os.path.join(current_dir, relative_path)


questions = defaultdict(list)

with open(data_path) as f:
    lines = f.read().splitlines()
    for line in lines:
        s = re.findall(r': (.+)', line)
        if s:
            section = s[0]
        else:
            questions[section].append(line.split(" "))

load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')
model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

countries = set()


for question in questions["capital-common-countries"]:
    countries.add(question[1])
    countries.add(question[3])
for question in questions["capital-world"]:
    countries.add(question[1])
    countries.add(question[3])

countries_vec = []
countries_label = []

for country in countries:
    countries_vec.append(model[country])
    countries_label.append(country)

countries_vec = np.array(countries_vec)

ward = linkage(countries_vec, method="ward")

plt.figure(figsize=(16, 8))

dendrogram(ward, labels=countries_label, leaf_rotation=90, leaf_font_size=8)

output_path = "../../outputs/ch06/result58.png"
output_path = os.path.join(current_dir, output_path)
plt.savefig(output_path)
