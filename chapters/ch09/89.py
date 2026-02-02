from dotenv import load_dotenv
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        super().__init__()
        self.sentences =sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inputs = self.sentences[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return len(self.sentences)

class ConvClassification(nn.Module):
    def __init__(self, input_size, output_size=1, batch_size=64, bert=None):
        super().__init__()
        self.bert = bert
        self.linear= nn.Linear(batch_size * 2, output_size)
        self.conv1 = nn.Conv2d(1, batch_size, kernel_size=(2, input_size))
        self.conv2 = nn.Conv2d(1, batch_size, kernel_size=(4, input_size))
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        embs = self.bert(**inputs)
        outputs = embs.last_hidden_state
        outputs = outputs.unsqueeze(1)
        out1 = F.relu(self.conv1(outputs))
        out2 = F.relu(self.conv2(outputs))

        out1 = F.max_pool2d(out1, kernel_size=(out1.size()[2], 1))
        out2 = F.max_pool2d(out2, kernel_size=(out2.size()[2], 1))

        out1 = out1.view(-1, self.batch_size)
        out2 = out2.view(-1, self.batch_size)

        outputs = torch.cat([out1, out2], dim=1)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)

        return outputs
    
def collate(batch):
    sentences, targets = list(zip(*batch))
    
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets

load_dotenv()

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"
SAVE_PATH = f"{PROJECT_ROOT}/data/ch09/sst2_ft.pt"

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)


train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")

train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist(), tokenizer)
dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

model = ConvClassification(input_size=768, output_size=2, batch_size=64, bert=bert_model).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def main():
    model.train()

    for batch in tqdm(train_dl):
        inputs, labels = batch 

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

    model.eval()
    total_acc = 0
    with torch.no_grad():
        for batch in tqdm(dev_dl):
            inputs, labels = batch 

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_acc += (preds == labels).float().mean().cpu()
    print(float(total_acc / len(dev_dl)))

if __name__ == "__main__":
    main()