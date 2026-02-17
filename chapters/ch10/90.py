from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax

MODEL_NAME = "openai-community/gpt2"
INPUT = "The movie was full of"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    inputs = tokenizer(INPUT, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        
        probs = softmax(logits)

        top = 10
        top_probs, top_ids = torch.topk(probs, top)

        for prob, token_id in zip(top_probs, top_ids):
            token = tokenizer.decode([token_id])
            print(f"{token} : {prob:.6f}")
            
if __name__ == "__main__":
    main()