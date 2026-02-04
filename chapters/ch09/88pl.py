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

INPUTS = ["The movie was full of incomprehensibilities.",
          "The movie was full of fun.",
          "The movie was full of excitement.",
          "The movie was full of crap.",
          "The movie was full of rubbish."]

class SST2Trainer(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_logs = []

    def forward(self, input):
        return self.model(**input)
    
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    

load_dotenv()

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
SST2_PATH = os.getenv('SST2_PATH')
MODEL_NAME = "google-bert/bert-base-uncased"

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

def main():

    LR = 1e-4

    trainer_model = SST2Trainer(model, lr=LR)


    ckpt = torch.load(f"{PROJECT_ROOT}/chapters/ch09/lightning_logs/version_5/checkpoints/epoch=0-step=264.ckpt", map_location="cpu")
    trainer_model.load_state_dict(ckpt["state_dict"], strict=True)
    print("load model")

    trainer_model.eval()

    with torch.no_grad():
        inputs = tokenizer(INPUTS, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = trainer_model(inputs)

        preds = torch.argmax(outputs.logits, dim=1)

    for s, pred in zip(INPUTS, preds):
        print(f"{s}\t{int(pred)}")

if __name__ == "__main__":
    main()