import torch
from q70 import emb, vector_size
from q72 import LogisticRegression, MyDataset, device
from q71 import train
from torch.utils.data import DataLoader
import torch.nn as nn 
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')

def main():
    epochs = 20
    lr = 0.001
    emb_ts = torch.tensor(emb)
    model = LogisticRegression(vector_size, 1, embedding=emb_ts).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    batch_size = 1

    train_ds = MyDataset(train)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    loss_hist = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, _, y in train_dl:
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

    save_path = PROJECT_ROOT + "/data/ch08/q73.pth"
    torch.save(model.state_dict(), save_path)
    print(f"save model")

if __name__ == "__main__":
    main()