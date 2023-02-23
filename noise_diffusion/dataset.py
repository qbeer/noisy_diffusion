import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as T

class NoisyMNIST(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, train=True, transform=None, std_min=0.01, std_max=1.0):
        if transform is None:
            transform = T.Compose([T.Resize(size=(32, 32)),  T.ToTensor()])
        self.dataset = MNIST(root='/tmp/', train=train, transform=transform, download=True)
        self.stds = torch.tensor([ 0.5995**i for i in range(10) ])

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        std_idx = torch.randint(0, 10, size=(1, ))
        std = self.stds[std_idx]
        noisy_x = x + torch.randn(x.size()) * std

        data = {
            "image": x,
            "label": y,
            "std": std,
            "noisy_image": noisy_x,
            "std_idx": std_idx
        }

        return data
