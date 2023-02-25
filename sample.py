from __future__ import annotations

import argparse

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from noise_diffusion.model import Unet

seed_everything(42, workers=True)


def run(args):
    model = Unet(
        backbone=args.backbone,
        in_chans=1 if args.dataset == 'mnist' else 3,
        num_classes=1 if args.dataset == 'mnist' else 3,
    )
    chkpt = torch.load(args.chkpt)
    model.load_state_dict(chkpt['state_dict'])
    model = model.to(device='cuda:0')
    model = model.eval()

    def run_sampling(T=50, L=10, sigma_1=1, sigma_10=0.01, eps=2e-5):
        samples = []
        x = torch.rand(
            size=(36, 1 if args.dataset == 'mnist' else 3, 32, 32),
            dtype=torch.float32,
        ).to(device='cuda:0')
        samples.append(x.cpu().permute(0, 2, 3, 1).detach().numpy())
        stds = torch.tensor(
            [sigma_1 * ((sigma_10/sigma_1)**(1/(L-1)))**i for i in range(L)],
        )
        for i in range(L):
            alpha = torch.tensor(eps).to(device='cuda:0') * \
                stds[i]**2 / stds[-1]**2
            i = torch.tensor([i], dtype=torch.long).repeat(36).view(36, 1)
            i = i.to(device='cuda:0')
            for _ in range(T):
                z = torch.randn(
                    size=(
                        36, 1 if args.dataset ==
                        'mnist' else 3, 32, 32,
                    ),
                ).to(device='cuda:0')
                x = x + alpha / 2. * \
                    model.forward(x, i) + torch.sqrt(alpha) / 2 * z  # trick?
                x = torch.clip(x, 0., 1.)
                samples.append(x.cpu().permute(0, 2, 3, 1).detach().numpy())
        return samples

    with torch.no_grad():
        samples = run_sampling()

    with imageio.get_writer(
        f'samples/{args.dataset}_model_samples.gif',
        mode='I', duration=0.02,
    ) as writer:
        for ind in tqdm(list(range(0, len(samples), 5))):
            sample = samples[ind]
            fig, axes = plt.subplots(
                6, 6, sharex=True, sharey=True, figsize=(15, 15),
            )
            axes = axes.flatten()
            for img, ax in zip(sample, axes):
                ax.imshow(img, vmin=0, vmax=1, interpolation=None)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            plt.savefig(f'/tmp/{ind}.png', dpi=50)
            plt.close()

            image = imageio.imread(f'/tmp/{ind}.png')
            writer.append_data(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=['mnist', 'cifar10'], default='mnist',
    )
    parser.add_argument(
        '--backbone', choices=['resnet50', 'resnet101', 'resnet152'],
        default='resnet50',
    )
    parser.add_argument('--chkpt', required=True, type=str)
    args = parser.parse_args()
    exit(run(args))
