import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, transform=True, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len # max len - special token(cls, sep)
        self.transform = transform

        self.x = pd.read_csv(path)["content"].apply(lambda x: tokenizer.encode(x))
        self.label = pd.read_csv(path)["lable"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        label = self.label.iloc[idx]
        if len(x) < self.max_len:
            x = x + [0]*(self.max_len-len(x))
        else:
            x = x[:self.max_len]

        pad_x = [1 if i >= 1 else 0 for i in x]
        if self.transform:
            x = torch.tensor(x)
            pad_x = torch.tensor(pad_x)
            label = torch.tensor(label)
        return x, pad_x, label
