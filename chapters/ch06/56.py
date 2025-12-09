import os
from dotenv import load_dotenv
import gensim
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../data/ch06/combined.csv"
data_path = os.path.join(current_dir, relative_path)

load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')
model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

def cos(x, y):
  return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

df = pd.read_csv(data_path)

ans = []
pred = []

for index, row in df.iterrows():
    ans.append(row["Human (mean)"])
    pred.append(cos(model[row["Word 1"]], model[row["Word 2"]]))

corr, p = spearmanr(ans, pred)

print(f"スピアマン相関係数：{corr}")