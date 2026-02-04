from dotenv import load_dotenv
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
import pytorch_lightning as pl

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

class SST2Trainer(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_logs = []

    def forward(self, input):
        output = self.model(**input)
        logit = output.logits
        prob = softmax(logit, dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch 

        labels = labels

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch 

        labels = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.val_logs.append((preds == labels).float().mean().item())

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_logs) > 0:
            acc = sum(self.val_logs) / len(self.val_logs)
            self.log("val_acc", acc, prog_bar=True)
            with open(f"{PROJECT_ROOT}/outputs/ch09/87pl.txt", "w") as f:
                f.write(f"{acc}")
            self.val_logs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    

load_dotenv()

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)


train = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")

train_ds = SST2Dataset(train["sentence"].tolist(), train["label"].tolist(), tokenizer)
dev_ds = SST2Dataset(dev["sentence"].tolist(), dev["label"].tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)


def main():

    MAX_EPOCHS = 1
    LR = 1e-4

    trainer_model = SST2Trainer(model, lr=LR)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
    )

    #学習
    trainer.fit(trainer_model, train_dl, dev_dl)

if __name__ == "__main__":
    main()