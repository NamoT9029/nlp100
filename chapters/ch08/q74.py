import torch
from q70 import emb, vector_size
from q72 import LogisticRegression, MyDataset, device
from q71 import dev
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')

def cal_acc(model, dev_dl, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, _, y in dev_dl:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            preds = (outputs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    save_path = PROJECT_ROOT + "/data/ch08/q73.pth"
    emb_ts = torch.tensor(emb)
    model = LogisticRegression(vector_size, 1, embedding=emb_ts).to(device)
    weight= torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(weight)
    batch_size = 1
    dev_ds = MyDataset(dev)
    dev_dl = DataLoader(dev_ds, batch_size, shuffle=False)

    print(cal_acc(model, dev_dl, device))

if __name__ == "__main__":
    main()