import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import UNet
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
from pathlib import Path
from dataset import LungDataset
from monai.losses import DiceFocalLoss

class TumorSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.loss_fn = DiceFocalLoss(
            sigmoid=True,
            include_background=True,
            gamma=2.0,
            lambda_dice=0.3,
            lambda_focal=1.7,
            reduction="mean"
        )

    def forward(self, x):
        return self.model(x)

    def _log_images(self, x, y, y_hat, stage):
        y_hat_np = y_hat[0][0].detach().cpu().numpy()

        fig = plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(x[0][0].cpu(), cmap='bone')
        plt.title('Input')

        plt.subplot(1, 4, 2)
        plt.imshow(y[0][0].cpu(), cmap='gray')
        plt.title('Ground Truth')

        plt.subplot(1, 4, 3)
        plt.imshow(y_hat_np, cmap='hot', vmin=0, vmax=1)
        plt.title('Raw Prediction (y_hat)')

        plt.subplot(1, 4, 4)
        plt.imshow(x[0][0].cpu(), cmap='bone')
        mask_overlay = np.ma.masked_where(y_hat_np < 0.2, y_hat_np)
        plt.imshow(mask_overlay, cmap='autumn', alpha=0.5)
        plt.title('Overlay (y_hat > 0.2)')

        self.logger.experiment.add_figure(f'{stage}_raw_logits', fig, self.global_step)
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)

        probs = torch.sigmoid(y_hat)
        preds = probs > 0.2

        if batch_idx % 100 == 0:
            self._log_images(x, y, y_hat, 'train')

        if self.global_step % 200 == 0:
            self.log("debug/pred_area", preds.sum() / preds.numel())
            self.log("debug/gt_area", y.sum() / y.numel())
            self.log("debug/diff_area", (preds.sum() - y.sum()) / y.numel())

        dice = (2 * (preds * y).sum()) / (preds.sum() + y.sum() + 1e-6)
        self.log("train_dice", dice)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        probs = torch.sigmoid(y_hat)
        preds = probs > 0.2

        if y.sum() > 0:
            dice = (2 * (preds * y).sum()) / (preds.sum() + y.sum() + 1e-6)
            self.log('val_dice', dice, prog_bar=True)

        if batch_idx % 20 == 0:
            self._log_images(x, y, y_hat, 'val')

        if self.global_step % 200 == 0:
            self.log("debug/pred_area", preds.sum() / preds.numel())
            self.log("debug/gt_area", y.sum() / y.numel())
            self.log("debug/diff_area", (preds.sum() - y.sum()) / y.numel())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),
        iaa.ElasticTransformation()
    ])

    path = Path("../datos/Preprocessed/train/")
    path_val = Path("../datos/Preprocessed/val/")
    train_dataset = LungDataset(path, augment_params=seq, tumor_oversampling_factor=8)
    val_dataset = LungDataset(path_val, augment_params=None)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    batch_size = 12
    num_workers = 8

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=num_workers, sampler=sampler, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False, pin_memory=True)

    model = TumorSegmentation()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        filename="tumor-{epoch:02d}-{val_dice:.4f}",
        dirpath="../weights"
    )

    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=25,
        min_delta=0.001,
        verbose=True
    )

    logger = TensorBoardLogger(save_dir="../logs", name="lung-tumor")

    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)
