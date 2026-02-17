from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from torch.nn.functional import softmax

MODEL_NAME = "openai-community/gpt2"
INPUT = "The movie was full of"
MAX_NEW_TOKENS = 16

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    inputs = tokenizer(INPUT, return_tensors="pt")
    input_ids = inputs.input_ids
    stop_token_id = tokenizer(".", return_tensors="pt").input_ids

    new_token_ids = []
    new_token_probs = []
    past_key_values = None
    
    model.eval()
    with torch.no_grad():
        while len(new_token_ids) < MAX_NEW_TOKENS and input_ids[0][0] != stop_token_id[0][0]:
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            
            logits = outputs.logits[0, -1, :]
            
            probs = softmax(logits, dim=-1)
            prob, token_id_tensor = torch.topk(probs, 1, dim=-1)

            next_token_id = token_id_tensor.item()
            next_prob = prob.item()
            
            new_token_ids.append(next_token_id)
            new_token_probs.append(next_prob)
            
            past_key_values = outputs.past_key_values
            input_ids = token_id_tensor.unsqueeze(0) 
    generated_text = tokenizer.decode(new_token_ids)
    print(f"output: {INPUT} |{generated_text}")
    print(f"probs: {', '.join(map(str, new_token_probs))}")

if __name__ == "__main__":
    main()