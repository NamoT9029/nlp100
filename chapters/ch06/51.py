import os
from dotenv import load_dotenv
import gensim
import numpy as np

load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')

model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)


def cos(x, y):
  return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(cos(model["United_States"], model["U.S."]))