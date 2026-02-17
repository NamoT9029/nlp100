from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm

load_dotenv()
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 64
BATCH_SIZE = 8

def make_prompt(inputs):
    content =   """
Classify each text below as strictly "Positive" or "Negative".
Output format: "Number. Label"

Texts:
"""
    
    for i, inpt in enumerate(inputs, 1):
        content = content+f"{i}. {inpt}\n"

    content = content + "\nOutputs:\n"
    return content
     
def refine_response(outputs):
    res = []

    for output in outputs:
        judge = output.split(". ")[1]

        if judge.lower() == "positive":
            res.append(1)
        elif judge.lower() == "negative":
            res.append(0)
        else:
            res.append(-1)
    return res

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    set_seed(42)

    dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")
    sentences = dev["sentence"].tolist()
    labels = dev["label"].tolist()

    all_batch_inputs = []
    raw_responses = []

    for i in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[i : i + BATCH_SIZE]

            prompt = make_prompt(batch_sentences)
            
            messages = [{"role": "user", "content": prompt}]
            
            input_ids = tokenizer.apply_chat_template(
                [messages], 
                return_tensors="pt", 
                add_generation_prompt=True
            )
            
            all_batch_inputs.append(input_ids)

    for batch in tqdm(all_batch_inputs):
        input_ids = batch.input_ids.to(device)

        outputs = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7)
        output = tokenizer.decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        raw_responses.extend(output.split("\n"))

    responses = refine_response(raw_responses)

    assert len(responses) == len(labels), f"出力数と入力数が一致しません\n出力 : {len(responses)}\n入力 : {len(labels)}"

    correct = 0

    for response, label in  zip(responses, labels):
        if response == label:
            correct += 1

    acc = correct / len(labels)
    print(f"acc: {acc}")

if __name__ == "__main__":
    main()