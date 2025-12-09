import os
from dotenv import load_dotenv
import gensim
import re
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../data/ch06/questions-words.txt"
data_path = os.path.join(current_dir, relative_path)

load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')
model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

questions = defaultdict(list)

with open(data_path) as f:
    lines = f.read().splitlines()
    for line in lines:
        s = re.findall(r': (.+)', line)
        if s:
            section = s[0]
        else:
            questions[section].append(line.split(" "))

ans = []

for question in questions["capital-common-countries"]:
    word, sim = model.most_similar(positive=[question[2], question[1]], negative=[question[0]],topn=1)[0]
    ans.append(question+[word, sim])

correct = 0

for w1, w2, w3, w4, w5, sim in ans:
    if w4 == w5:
        correct += 1

acc = correct / len(ans)

print(f"意味的アナロジーの正解率：{acc}")

