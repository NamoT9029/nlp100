from transformers import AutoTokenizer
from pprint import pprint

MODEL_NAME = "google-bert/bert-base-uncased"
INPUT = "The movie was full of incomprehensibilities."
MAX_LENGTH = 64

def main():    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    tokens = tokenizer(INPUT,
                      max_length=MAX_LENGTH,
                      return_tensors="pt")

    pprint(tokens)

if __name__ == "__main__":
    main()