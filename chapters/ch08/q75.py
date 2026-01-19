from torch.nn.utils.rnn import pad_sequence
import torch
from q71 import train


def collate(batch):
    input_ids, _, targets = list(zip(*batch))
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    targets = torch.stack(targets)

    return input_ids, targets

def main():
    print(collate([[data["input_ids"], data["text"], data["label"]] for data in train[:3]]))

if __name__ == "__main__":
    main()
    