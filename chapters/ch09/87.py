from dotenv import load_dotenv
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch
from tqdm import tqdm
from torch.nn.functional import softmax

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        super().__init__()
        self.sentences =sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inputs = self.sentences[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, label

    def __len__(self):
        return len(self.sentences)

def collate(batch):
    sentences, targets = list(zip(*batch))
    
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    targets = torch.stack(targets)

    return inputs, targets

load_dotenv()

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"
SAVE_PATH = f"{PROJECT_ROOT}/data/ch09/sst2_ft.pt"

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)


train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")

train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist(), tokenizer)
dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def main():
    model.train()

    for batch in tqdm(train_dl):
        inputs, labels = batch 

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), SAVE_PATH)
    print("save model")

    model.eval()
    total_acc = 0
    with torch.no_grad():
        for batch in tqdm(dev_dl):
            inputs, labels = batch 

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_acc += (preds == labels).float().mean().cpu()

    print(float(total_acc / len(dev_dl)))

if __name__ == "__main__":
    main()