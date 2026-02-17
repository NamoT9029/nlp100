from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT = "What do you call a sweet eaten after dinner?"
MAX_NEW_TOKENS = 64

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    set_seed(42)

    messages = [
        {"role":"user", "content":INPUT}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").input_ids

    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7)

        output = tokenizer.decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]


    print(f"prompt:\n{text}")
    print(f"input:\n{INPUT}\n")
    print(f"output: {output}")

if __name__ == "__main__":
    main()

