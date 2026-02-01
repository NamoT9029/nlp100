from dotenv import load_dotenv
import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        super().__init__()
        self.sentences =sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inputs = torch.tensor(self.tokenizer(self.sentences[index], return_tensor="pt")["input_ids"])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return inputs, label

    def __len__(self):
        return len(self.sentences)

load_dotenv()

SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")
train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist(), tokenizer)
dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist(), tokenizer)


def main():
    print(train_ds[0])
    print(dev_ds[0])

if __name__ == "__main__":
    main()