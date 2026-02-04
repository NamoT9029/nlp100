from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn.functional import softmax
# from transformers import pipeline

MODEL_NAME = "google-bert/bert-base-uncased"
INPUT = "The movie was full of [MASK]."
MAX_LENGTH = 64

def main():    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    inputs = tokenizer(INPUT,
                      max_length=MAX_LENGTH,
                      return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_token_id = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        mask_logits = logits[0, mask_token_id, :]
        
        probs = softmax(mask_logits, dim=-1)

        top = 10
        top_probs, top_indices = torch.topk(probs, top, dim=-1)

        for i in range(top):
            token = tokenizer.decode(top_indices[0, i])
            print(token, float(top_probs[0, i]))
            
if __name__ == "__main__":
    main()