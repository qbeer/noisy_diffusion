import torch
import matplotlib.pyplot as plt
from noise_diffusion.model import Unet

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

def run():
    model = Unet.load_from_checkpoint('lightning_logs/noisy_diffusion/epoch=245-valid_loss=52.303.ckpt')
    model = model.to(device='cuda:0')
    model = model.eval()

    def sample(T=100, L=10, sigma_1=1, sigma_10=0.01, eps=2e-5):
        x = torch.rand(size=(16, 1, 32, 32), dtype=torch.float32).to(device='cuda:0')
        stds = torch.tensor([ sigma_1 * ( (sigma_10/sigma_1)**(1/(L-1)) )**i for i in range(L) ])
        for i in range(L):
            alpha = torch.tensor(eps).to(device='cuda:0') * stds[i]**2 / stds[-1]**2 
            i = torch.tensor([i], dtype=torch.long).repeat(16).view(16, 1)
            i = i.to(device='cuda:0')
            for _ in range(T):
                z = torch.randn(size=(16, 1, 32, 32)).to(device='cuda:0')
                x = x + alpha / 2. * model.forward(x, i) + torch.sqrt(alpha) * z
                x = torch.clip(x, 0., 1.)
        return x.cpu().detach().numpy()

    with torch.no_grad():
        samples = sample()
    
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(16, 16))
    axes = axes.flatten()
    for img, ax in zip(samples, axes):
        img = img.reshape(32, 32)
        ax.imshow(img, vmin=0, vmax=1, interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()
    plt.savefig('samples/model_samples.png', dpi=100)

if __name__ == "__main__":
    exit(run())
