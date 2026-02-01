from dotenv import load_dotenv
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
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

def collate(batch):
    input_ids, targets = list(zip(*batch))
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    targets = torch.stack(targets)

    return input_ids, targets


load_dotenv()

SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")

train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist(), tokenizer)
dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate,num_workers=2)
dev_dl = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)

def main():
    for batch in train_dl:
        print(batch)
        break

if __name__ == "__main__":
    main()