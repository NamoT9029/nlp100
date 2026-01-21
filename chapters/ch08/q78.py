import torch
from q70 import emb, vector_size
from q72 import LogisticRegression, MyDataset, device
from q71 import train, dev
from q75 import collate
from torch.utils.data import DataLoader
import torch.nn as nn 
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')

def cal_acc(model, dev_dl, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in dev_dl:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            preds = (outputs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    epochs = 20
    lr = 0.01
    emb_ts = torch.tensor(emb)
    model = LogisticRegression(vector_size, 1, embedding=emb_ts, freeze=False).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    batch_size = 256
    
    model.train()
    train_ds = MyDataset(train)
    dev_ds = MyDataset(dev)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    loss_hist = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1} Loss {avg_loss:4f}")
        if epoch > 1:
            if avg_loss > loss_hist[epoch-1]:
                break
        loss_hist.append(avg_loss)
    print(cal_acc(model, dev_dl, device))
if __name__ == "__main__":
    main()