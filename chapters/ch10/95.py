from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from pprint import pprint

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUTS = ["What do you call a sweet eaten after dinner?",
          "Please give me the plural form of the word with its spelling in reverse order."]

MAX_NEW_TOKENS = 128

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    set_seed(42)

    messages = []
    res = []
    model.eval()
    with torch.no_grad():
        for INPUT in INPUTS:
            messages.append({"role":"user", "content":INPUT})

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer([text], return_tensors="pt").input_ids

            outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7)

            output = tokenizer.decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
            res.append(output)

            messages.append({"role": "assistant", "content": output})

    print("prompt: ")
    pprint(text)
    for INPUT, OUTPUT in zip(INPUTS, res):
        print(f"\ninput: {INPUT}")
        print(f"\noutput: {OUTPUT}")

if __name__ == "__main__":
    main()

