from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch

load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
MODEL_NAME = "google-bert/bert-base-uncased"
SAVE_PATH = f"{PROJECT_ROOT}/data/ch09/sst2_ft.pt"
INPUTS = ["The movie was full of incomprehensibilities.",
          "The movie was full of fun.",
          "The movie was full of excitement.",
          "The movie was full of crap.",
          "The movie was full of rubbish."]

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

def main():
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    print("load model")

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(INPUTS, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)

    for s, pred in zip(INPUTS, preds):
        print(f"{s}\t{int(pred)}")
        
if __name__ == "__main__":
    main()