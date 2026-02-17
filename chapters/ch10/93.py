from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
import math

MODEL_NAME = "openai-community/gpt2"
INPUTS = ["The movie was full of surprises",
          "The movies were full of surprises",
          "The movie were full of surprises",
          "The movies was full of surprises"]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    p = []

    model.eval()
    with torch.no_grad():
        for INPUT in INPUTS:
            input_ids = tokenizer(INPUT, return_tensors="pt").input_ids
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            
            loss = outputs.loss
        
            p.append(math.exp(loss.item()))

    for text, score in zip(INPUTS, p):
        print(f"text: {text}\tperplexity: {score}")
if __name__ == "__main__":
    main()