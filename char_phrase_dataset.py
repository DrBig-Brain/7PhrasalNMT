import torch
from torch.utils.data import Dataset

class CharPhraseDataset(Dataset):
    def __init__(self, x, y, phrases, sequence_len, ch2i, phrase2idx):
        self.x, self.y, self.phrases = x, y, phrases
        self.sequence_len = sequence_len
        self.ch2i = ch2i
        self.phrase2idx = phrase2idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = [self.ch2i.get(c, 0) for c in self.x[idx]]
        y = [self.ch2i.get(c, 0) for c in self.y[idx]]
        p = [self.phrase2idx.get(tag, 0) for tag in self.phrases[idx]]
        x = x[:self.sequence_len] + [0]*(self.sequence_len - len(x))
        y = y[:self.sequence_len] + [0]*(self.sequence_len - len(y))
        p = p[:self.sequence_len] + [0]*(self.sequence_len - len(p))
        return torch.tensor(x), torch.tensor(y), torch.tensor(p)
