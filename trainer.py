import torch
from torch.utils.data import DataLoader

class TrainerConfig:
    def __init__(self, max_epochs=10, batch_size=64, learning_rate=3e-4, device='cpu'):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y, p in self.dataloader:
            x, y, p = x.to(self.config.device), y.to(self.config.device), p.to(self.config.device)
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            out = self.model(x, tgt_in, p, p[:, :-1])
            loss = self.criterion(out.reshape(-1, out.size(-1)), tgt_out.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def train(self):
        for epoch in range(1, self.config.max_epochs + 1):
            loss = self.train_epoch()
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
