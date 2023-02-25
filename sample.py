from __future__ import annotations

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from noise_diffusion.model import Unet

seed_everything(42, workers=True)


def run():
    model = Unet.load_from_checkpoint(
        'lightning_logs/noisy_diffusion/epoch=97-valid_loss=48.022.ckpt',
    )
    model = model.to(device='cuda:0')
    model = model.eval()

    def run_sampling(T=100, L=10, sigma_1=1, sigma_10=0.01, eps=2e-5):
        samples = []

        x = torch.rand(
            size=(16, 1, 32, 32),
            dtype=torch.float32,
        ).to(device='cuda:0')
        samples.append(x.cpu().detach().numpy())
        stds = torch.tensor(
            [sigma_1 * ((sigma_10/sigma_1)**(1/(L-1)))**i for i in range(L)],
        )
        for i in range(L):
            alpha = torch.tensor(eps).to(device='cuda:0') * \
                stds[i]**2 / stds[-1]**2
            i = torch.tensor([i], dtype=torch.long).repeat(16).view(16, 1)
            i = i.to(device='cuda:0')
            for _ in range(T):
                z = torch.randn(size=(16, 1, 32, 32)).to(device='cuda:0')
                x = x + alpha / 2. * \
                    model.forward(x, i) + torch.sqrt(alpha) / 2 * z  # trick?
                x = torch.clip(x, 0., 1.)
                samples.append(x.cpu().detach().numpy())
        return samples

    with torch.no_grad():
        samples = run_sampling()

    with imageio.get_writer(
        'samples/model_samples.gif',
        mode='I', duration=0.02,
    ) as writer:
        for ind in tqdm(list(range(0, len(samples), 3))):
            sample = samples[ind]
            fig, axes = plt.subplots(
                4, 4, sharex=True, sharey=True, figsize=(12, 12),
            )
            axes = axes.flatten()
            for img, ax in zip(sample, axes):
                img = img.reshape(32, 32)
                ax.imshow(img, vmin=0, vmax=1, interpolation=None)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            plt.savefig(f'/tmp/{ind}.png', dpi=15)
            plt.close()

            image = imageio.imread(f'/tmp/{ind}.png')
            writer.append_data(image)


if __name__ == '__main__':
    exit(run())
