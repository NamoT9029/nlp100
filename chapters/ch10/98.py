import os
import sys
import pandas as pd
import torch
import warnings
from pprint import pprint
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
load_dotenv()

SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "openai-community/gpt2"
MAX_LENGTH = 128
BATCH_SIZE = 64


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def make_dataset(sentences, labels):
    all_data = []
    for sent, int_label in zip(sentences, labels):
        label = "Positive" if int_label == 1 else "Negative"
        full_text = f"Classify text below as strictly \"Positive\" or \"Negative\".\n\nInput: {sent}\n\nLabel: {label}"
        all_data.append({"text": full_text})
    return Dataset.from_list(all_data)

def preprocess_function(ex):
    tokenized = tokenizer(
        ex["text"],
        padding="max_length",  
        truncation=True,       
        max_length=MAX_LENGTH,
        return_tensors="pt"    
    )
    tokenized["labels"] = tokenizer(
        ex["text"],
        padding="max_length",  
        truncation=True,       
        max_length=MAX_LENGTH,
        return_tensors="pt"  
    )["input_ids"]

    return tokenized

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    preds = [1 if "positive" in pred.split("Label:")[-1].lower() else 0 for pred in preds]
    labels = [1 if "positive" in label.split("Label:")[-1].lower() else 0 for label in labels]

    assert len(preds) == len(labels), f"出力数と入力数が一致しません\n出力 : {len(preds)}\n入力 : {len(labels)}"

    cnt = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            cnt += 1
    return {"accuracy" : cnt / len(preds)}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def main():

    train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t", on_bad_lines='skip')
    dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t", on_bad_lines='skip')

    train_dataset = make_dataset(train["sentence"].tolist(), train["label"].tolist())
    dev_dataset = make_dataset(dev["sentence"].tolist(), dev["label"].tolist())

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="./gpt2-ft",
        per_device_train_batch_size=BATCH_SIZE, 
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-5,
        fp16=True
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    pprint(trainer.evaluate())

if __name__ == "__main__":
    main()