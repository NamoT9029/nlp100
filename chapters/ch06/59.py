import os
from dotenv import load_dotenv
import gensim
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans

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

tsne = TSNE(n_components=2, random_state=42, max_iter=10000, metric="cosine")

preds_tsne = tsne.fit_transform(countries_vec)

plt.figure(figsize=(10, 10))
sns.scatterplot(x=preds_tsne[:,0], y=preds_tsne[:,1])

output_path = "../../outputs/ch06/result59.png"
output_path = os.path.join(current_dir, output_path)
plt.savefig(output_path)



plt.figure(figsize=(10, 10))
km = KMeans(n_clusters=5, random_state=42)
preds_km = km.fit_predict(countries_vec)

for label in range(5):
    idx = preds_km == label
    plt.scatter(preds_tsne[idx, 0], preds_tsne[idx, 1], label=label)
output_path = "../../outputs/ch06/result59_kmeans.png"
output_path = os.path.join(current_dir, output_path)
plt.savefig(output_path)


plt.figure(figsize=(10, 10))
preds_ward = linkage(countries_vec, method="ward")
ward = fcluster(preds_ward, t=5, criterion="maxclust") 
for label in range(5):
    idx = ward == (label+1)
    plt.scatter(preds_tsne[idx, 0], preds_tsne[idx, 1], label=label)
output_path = "../../outputs/ch06/result59_ward.png"
output_path = os.path.join(current_dir, output_path)
plt.savefig(output_path)