import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

from noise_diffusion.model import Unet
from noise_diffusion.dataset import NoisyMNIST

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

def train(args):
    model = Unet()

    train_ds = NoisyMNIST(train=True)
    val_ds = NoisyMNIST(train=False)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=5,
        pin_memory=False)

    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=5,
        pin_memory=False)

    wandb_logger = WandbLogger(project='noise-diffusion', entity='elte-ai4covid', 
        save_dir='./lightning_logs/')

    class LogPredictionsCallback(Callback):
        def _sample(self, pl_module, T=100, L=10, eps=2e-5):
            stds = torch.tensor([ 0.5995**i for i in range(L) ]).to(device='cuda:0')
            x = torch.rand(size=(16, 1, 32, 32)).to(device='cuda:0')
            for i in range(L):
                alpha = torch.tensor(eps).to(device='cuda:0') * stds[i]**2 / stds[-1]**2 
                for _ in range(T):
                    z = torch.randn(size=(16, 1, 32, 32)).to(device='cuda:0')
                    x = x + alpha / 2. * pl_module(x, stds[i]) + torch.sqrt(alpha) * z
            return x.cpu().detach().numpy()


        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            """Called when the validation batch ends."""
            if batch_idx == 0:
                samples = self._sample(pl_module)
                wandb_logger.log_image(key='sample_images', images=[sample[0] for sample in samples])

    log_pred_callback = LogPredictionsCallback()

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='./lightning_logs/binary_mammo/',
        save_top_k=3,
        mode='min',
        filename='{epoch:02d}-{valid_loss:.3f}')
    
    trainer = pl.Trainer(max_epochs=args.epochs,
                         devices=1,
                         accelerator='gpu',
                         logger=wandb_logger,
                         val_check_interval=1.0,
                         min_epochs=args.epochs,
                         callbacks=[
                            checkpoint_callback,
                            log_pred_callback
                         ],
                         deterministic=True)

    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10, required=False)

    args = parser.parse_args()
    train(args)
