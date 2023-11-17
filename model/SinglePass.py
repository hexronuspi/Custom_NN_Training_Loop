import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import tqdm

class SinglePass(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self._model_state = None
        self._train_state = None
        self.device = device
        self.clip_grad_norm = None
        self.metrics = {"train": {}, "test": {}}
        self.using_tpu = False
        self.fp16 = None

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fetch_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)

    def fetch_scheduler(self):
        return CosineAnnealingLR(self.fetch_optimizer(), T_max=10)

    def train_one_step(self, data, target, optimizer, scheduler, device):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = self(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss 
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss

    def train_one_epoch(self, data_loader, optimizer, scheduler, device):
        self.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc='Training', total=len(data_loader))
        for data, target in progress_bar:
            loss = self.train_one_step(data, target, optimizer, scheduler, device)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/data.size(0))})
        train_loss = epoch_loss / len(data_loader)
        print(f'Train Loss: {train_loss:.4f}')
        return train_loss

    def test(self, test_dataset, batch_size, device):
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
    
    def fit(self, train_dataset, test_dataset, batch_size, epochs, device):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler()

        if next(self.parameters()).device != device:
            self.to(device)

        for epoch in range(epochs):
            print(f'Epoch: {epoch+1}/{epochs}')
            train_loss = self.train_one_epoch(train_loader, optimizer, scheduler, device)
            self.metrics["train"][f"epoch_{epoch+1}"] = train_loss

            # Test after each epoch
            self.test(test_dataset, batch_size, device)
            optimizer.step()
            scheduler.step()
