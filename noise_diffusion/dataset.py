import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as T

class NoisyMNIST(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, train=True, std_10=0.01, std_1=1.0, L=10):
        if train:
            transform = T.Compose([T.Resize(size=(32, 32)),
                                   T.RandomApply([T.RandomCrop(size=(28, 28))], p=0.5),
                                   T.RandomEqualize(p=0.5),
                                   T.RandomAutocontrast(p=0.5),
                                   T.RandomApply([T.RandomRotation(degrees=(-5, 5))], p=0.5),
                                   T.Resize(size=(32, 32)),
                                   T.ToTensor()])
        else:
            transform = T.Compose([T.Resize(size=(32, 32)), T.ToTensor()])
            
        self.dataset = MNIST(root='/tmp/', train=train, transform=transform, download=True)
        self.stds = torch.tensor([ std_1 * ( (std_10/std_1)**(1/(L-1)) )**i for i in range(L) ])

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        std_idx = torch.randint(0, 10, size=(1, ))
        std = self.stds[std_idx]
        x = torch.clip(x, 0, 1)
        noisy_x = x + torch.randn(x.size()) * std
        noisy_x = torch.clip(noisy_x, 0, 1)

        data = {
            "image": x,
            "label": y,
            "std": std,
            "noisy_image": noisy_x,
            "std_idx": std_idx
        }

        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from pytorch_lightning import seed_everything

    seed_everything(42, workers=True)
    
    train_ds = NoisyMNIST()
    samples = random.choices(train_ds, k=9)
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        ax.imshow(sample['noisy_image'][0].numpy(), vmin=0, vmax=1)
        ax.set_title(f"Label : {sample['label']}, noise : {sample['std'].numpy()[0]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('samples/noisy_samples.png')
    plt.close()

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        ax.imshow(sample['image'][0].numpy(), vmin=0, vmax=1)
        ax.set_title(f"Label : {sample['label']}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('samples/samples.png')