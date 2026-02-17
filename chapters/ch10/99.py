import os
import sys
import pandas as pd
import torch
import warnings
from pprint import pprint
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
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
tokenizer.padding_side = "left"

def make_dataset(sentences, labels):
    all_data = []
    for sent, int_label in zip(sentences, labels):
        chosen_label = "Positive" if int_label == 1 else "Negative"
        rejected_label = "Positive" if int_label == 0 else "Negative"
        full_text = f"Classify text below as strictly \"Positive\" or \"Negative\".\n\nInput: {sent}\n\nLabel: "
        all_data.append({"prompt": full_text, "chosen": chosen_label + tokenizer.eos_token , "rejected":rejected_label + tokenizer.eos_token})
    return Dataset.from_list(all_data)



def main():

    train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t", on_bad_lines='skip')
    dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t", on_bad_lines='skip')

    train_dataset = make_dataset(train["sentence"].tolist(), train["label"].tolist())
    dev_dataset = make_dataset(dev["sentence"].tolist(), dev["label"].tolist())

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],
    )

    dpo_config = DPOConfig(
        output_dir="./gpt2-dpo",
        bf16=True,
        max_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        save_strategy="epoch",
        eval_strategy="epoch",
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
        save_total_limit=1
    )

    dpo_trainer = DPOTrainer(
        model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    dpo_trainer.train()

    pprint(dpo_trainer.evaluate())

if __name__ == "__main__":
    main()