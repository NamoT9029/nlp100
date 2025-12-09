import os
from dotenv import load_dotenv
import gensim
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import re

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
km = KMeans(n_clusters=5, random_state=42)
preds = km.fit_predict(countries_vec)

for label, pred in zip(countries_label, preds):
    print(label, pred)