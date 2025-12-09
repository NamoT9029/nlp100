import os
from dotenv import load_dotenv
import gensim

load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')

model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

for sim in model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10):
    print(*sim)