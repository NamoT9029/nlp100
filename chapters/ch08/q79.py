import torch
from q70 import emb, vector_size
from q72 import MyDataset, device
from q71 import train, dev
from q75 import collate
from torch.utils.data import DataLoader
import torch.nn as nn 
from dotenv import load_dotenv
import os
import torch.nn.functional as F

load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')

def cal_acc(model, dev_dl, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dev_dl:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            preds = (outputs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, embedding=None, freeze=True):
        super().__init__()
        self.linear= nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, (h, c) = self.lstm(emb)
        x = h[-1]
        y = self.linear(x)
        z = self.sigmoid(y)
        return z

class CNN(nn.Module):
    def __init__(self, input_size, output_size=1, batch_size=64, embedding=None, freeze=True):
        super().__init__()
        self.linear= nn.Linear(batch_size * 3, output_size)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.conv1 = nn.Conv2d(1, batch_size, kernel_size=(2, input_size))
        self.conv2 = nn.Conv2d(1, batch_size, kernel_size=(4, input_size))
        self.conv3 = nn.Conv2d(1, batch_size, kernel_size=(8, input_size))
        self.batch_size = batch_size

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out = emb.unsqueeze(1)

        out1 = F.relu(self.conv1(out))
        out2 = F.relu(self.conv2(out))
        out3 = F.relu(self.conv3(out))
        
        out1 = F.max_pool2d(out1, kernel_size=(out1.size()[2], 1))
        out2 = F.max_pool2d(out2, kernel_size=(out2.size()[2], 1))
        out3 = F.max_pool2d(out3, kernel_size=(out3.size()[2], 1))
        
        out1 = out1.view(-1, self.batch_size)
        out2 = out2.view(-1, self.batch_size)
        out3 = out3.view(-1, self.batch_size)

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.linear(out)
        out = self.sigmoid(out)

        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, embedding=None, freeze=True):
        super().__init__()
        self.linear= nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, h = self.rnn(emb)
        x = h[-1]
        y = self.linear(x)
        z = self.sigmoid(y)
        return z


def main():
    epochs = 20
    lr = 0.01
    emb_ts = torch.tensor(emb)
    batch_size = 256
    # model = LSTM(input_size=vector_size, hidden_size=128, output_size=1, embedding=emb_ts, freeze=False).to(device)
    # model = CNN(input_size=vector_size, output_size=1, batch_size=batch_size, embedding=emb_ts, freeze=False).to(device)
    model = RNN(input_size=vector_size, hidden_size=128, output_size=1, embedding=emb_ts, freeze=False).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

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