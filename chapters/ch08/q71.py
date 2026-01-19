from dotenv import load_dotenv
import os
import pandas as pd
import torch
from q70 import token_to_idx 

load_dotenv()
SST2_PATH = os.getenv('SST2_PATH')


train_data = pd.read_csv(f"{SST2_PATH}/train.tsv", sep="\t")
dev_data = pd.read_csv(f"{SST2_PATH}/dev.tsv", sep="\t")


def create_dict_objects(df):
    dict_objects = []
    for _, row in df.iterrows():
        dict_object = dict()
        dict_object["text"] = row["sentence"]
        dict_object["label"] = torch.tensor([row["label"]], dtype=torch.float32)
        dict_object["input_ids"] = torch.tensor([token_to_idx[word] for word in row["sentence"].split() if token_to_idx.get(word, False)])
        if len(dict_object["input_ids"]) != 0:
            dict_objects.append(dict_object)
    return dict_objects

train = create_dict_objects(train_data)
dev = create_dict_objects(dev_data)

def main():
    print(f"train_data: {len(train_data)}")
    print(f"dev_data: {len(dev_data)}")
    print(f"train: {len(train)}")
    print(f"dev: {len(dev)}")
    print(train[0])

if __name__ == "__main__":
    main()
    