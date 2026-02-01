from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
# from transformers import pipeline

MODEL_NAME = "google-bert/bert-base-uncased"
INPUTS = ["The movie was full of fun.",
          "The movie was full of excitement.",
          "The movie was full of crap.",
          "The movie was full of rubbish."]

MAX_LENGTH = 64


def main():    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embs = []

    CLS_INPUTS = [f"[CLS] {s}" for s in INPUTS]

    with torch.no_grad():
        for cls_input in CLS_INPUTS:
            inputs = tokenizer(cls_input,
                            max_length=MAX_LENGTH,
                            return_tensors="pt")

            outputs = model(**inputs)
            embs.append(outputs.last_hidden_state[:, 0, :])

    for i in range(len(INPUTS)):
        for j in range(i+1, len(INPUTS)):
            sim = cosine_similarity(embs[i], embs[j], dim=1)
            print(f"{INPUTS[i]}\t{INPUTS[j]}\t: {sim}")

if __name__ == "__main__":
    main()