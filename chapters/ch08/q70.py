from dotenv import load_dotenv
import gensim
import os
import numpy as np
load_dotenv()
MODEL_PATH= os.getenv('WORD2VEC_MODEL_PATH')

w2v = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

vector_size = w2v.vector_size
words = w2v.index_to_key

emb = np.zeros((len(words)+1, vector_size), dtype="float32")

emb[1:] = w2v.get_normed_vectors()

idx_to_token = {0:"<PAD>"}
token_to_idx = {"<PAD>":0}
for i, word in enumerate(words, 1):
    idx_to_token[i] = word
    token_to_idx[word] = i

def main():
    print(emb.shape)
    print(len(idx_to_token))
    print(len(token_to_idx))
    print("=============")
    print(emb[923])
    print(idx_to_token[923])
    print(token_to_idx["Japan"])

if __name__ == "__main__":
    main()
    