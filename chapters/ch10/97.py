from transformers import AutoTokenizer, AutoModel, set_seed
import torch
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn

load_dotenv()
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "openai-community/gpt2"
MAX_NEW_TOKENS = 64
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
MAX_EPOCHS = 1
LR = 1e-4

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels):
        super().__init__()
        self.sentences =sentences
        self.labels = labels

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

class SST2Trainer(pl.LightningModule):
    def __init__(self, model, lr=1e-4, num_labels=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_logs = []
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss() 

    def forward(self, input):
        outputs = self.model(**input)
        last_hidden_state = outputs.last_hidden_state[:,-1,:]
        logits = self.classifier(last_hidden_state)
        return logits

    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch 

        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch 

        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.val_logs.append((preds == labels).float().mean().item())

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_logs) > 0:
            acc = sum(self.val_logs) / len(self.val_logs)
            self.log("val_acc", acc, prog_bar=True)
            with open(f"{PROJECT_ROOT}/outputs/ch10/97.txt", "w") as f:
                f.write(f"{acc}")
            self.val_logs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModel.from_pretrained(MODEL_NAME)
set_seed(42)

def main():
    train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
    dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")

    train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist())
    dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist())

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)

    trainer_model = SST2Trainer(model, lr=LR)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
    )

    trainer.fit(trainer_model, train_dl, dev_dl)

if __name__ == "__main__":
    main()