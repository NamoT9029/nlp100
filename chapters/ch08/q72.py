import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size, embedding, freeze=True):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        x = torch.mean(emb, dim=1)
        y = self.layer(x)
        z = self.sigmoid(y)
        return z

class MyDataset(Dataset):
    def __init__(self, dict_objects):
        super().__init__()
        self.dict_objects = dict_objects

    def __getitem__(self, index):
        input_ids = self.dict_objects[index]["input_ids"]
        text = self.dict_objects[index]["text"]
        label = self.dict_objects[index]["label"]
        return input_ids, text, label
    
    def __len__(self):
        return len(self.dict_objects)