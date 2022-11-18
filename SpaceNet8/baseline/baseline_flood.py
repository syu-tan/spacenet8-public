"""
from `SpaceNet8/baseline/baseline_flood.ipynb`
"""

from utils.utils import TimmUnet
from baseline_foundation import (
    SpaceNnet8Dataset, SpaceNnet8Module,
    MultiBCEDiceLoss, MultiBCETverskyLoss, BCEDiceLoss, DiceLoss,
    FocalDiceLoss, SoftBCEDiceLoss, BCELovaszLoss,
    mixup,
    DualHSV, PostFog
)
from typing import List, Set, Dict, Any
import os
import warnings
import random
from pprint import pprint
import copy
from typing import List, Tuple
import glob
import json
import csv
# import dataclasses
from joblib import Parallel, delayed

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from box import Box
import matplotlib.pyplot as plt

import tifffile
from osgeo import gdal

from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from adabelief_pytorch import AdaBelief
import segmentation_models_pytorch as smp

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import LightningDataModule, LightningModule

import wandb
wandb.login(key='****')

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
pd.options.display.max_colwidth = 250
pd.options.display.max_rows = 30


class SpaceNet8Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.model.loss)

    def __build_model(self):
        if self.cfg.model.architecture == 'smp':
            self.backbone = smp.UnetPlusPlus(encoder_name=cfg.model.encoder_name,
                                             encoder_weights="imagenet",
                                             decoder_attention_type='scse',
                                             in_channels=cfg.model.in_channels, activation=cfg.model.act,
                                             decoder_channels=cfg.model.decoder_channels,
                                             decoder_use_batchnorm=cfg.model.decoder_use_batchnorm,
                                             classes=cfg.model.out_channels)
        elif self.cfg.model.architecture == 'unet':
            self.backbone = smp.Unet(
                in_channels=cfg.model.in_channels,
                classes=cfg.model.out_channels,
                **cfg.model.unet_params)
        elif self.cfg.model.architecture == 'timmu':
            self.backbone = TimmUnet(
                in_chans=cfg.model.in_channels,
                out_chans=cfg.model.out_channels,
                pretrained=True,
                channels_last=False,
                **self.cfg.model.timmu.encoder_params,
            )

        else:
            raise ValueError(
                f'Check `cfg.model.architecture` >>> {self.model.architecture}')

    def forward(self, x):
        feat = self.backbone(x)
        return feat

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, 'val')

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float()
        images = images.float()

        if torch.rand(1)[0] < self.cfg.mixup and mode == 'train':
            mix_images, target_a, target_b, lam = mixup(
                images, labels, alpha=0.5)
            outputs = self(mix_images)
            loss = self._criterion(outputs, target_a[:, 10:14, ...]) * lam + \
                (1 - lam) * self._criterion(outputs, target_b[:, 10:14, ...])

        else:
            outputs = self(images)
            loss = self._criterion(outputs, labels[:, 10:14, ...])

        logits = torch.sigmoid(outputs)
        preds = (logits > cfg.model.threshold).float()
        return_dict = {'loss': loss, }

        # metrics
        for c in range(4):
            preds_c, labels_c = preds[:, c, :, :], labels[:, c, :, :]
            tp = (preds_c * labels_c).sum().to(torch.float32)
            tn = ((1. - preds_c) * (1. - labels_c)).sum().to(torch.float32)
            fp = (preds_c * (1. - labels_c)).sum().to(torch.float32)
            fn = ((1. - preds_c) * labels_c).sum().to(torch.float32)
            return_dict[f'TP_{c}'] = tp.unsqueeze(dim=0).detach().cpu()
            return_dict[f'TN_{c}'] = tn.unsqueeze(dim=0).detach().cpu()
            return_dict[f'FP_{c}'] = fp.unsqueeze(dim=0).detach().cpu()
            return_dict[f'FN_{c}'] = fn.unsqueeze(dim=0).detach().cpu()

            precision = tp / (tp + fp + cfg.eps)
            recall = tp / (tp + fn + cfg.eps)
            f1 = 2 * (precision*recall) / (precision + recall + cfg.eps)
            iou = tp / (tp + fp + fn + cfg.eps)

            return_dict[f'Precision_{c}'] = precision.unsqueeze(
                dim=0).detach().cpu()
            return_dict[f'Recall_{c}'] = recall.unsqueeze(dim=0).detach().cpu()
            return_dict[f'F1_{c}'] = f1.unsqueeze(dim=0).detach().cpu()
            return_dict[f'IoU_{c}'] = iou.unsqueeze(dim=0).detach().cpu()

            # logging
            self.log(f'{mode}/iter_TP_{c}', tp)
            self.log(f'{mode}/iter_TN_{c}', tn)
            self.log(f'{mode}/iter_FP_{c}', fp)
            self.log(f'{mode}/iter_FN_{c}', fn)

            self.log(f'{mode}/iter_Precision_{c}', precision)
            self.log(f'{mode}/iter_Recall_{c}', recall)

            self.log(f'{mode}/iter_F1_{c}', f1)
            self.log(f'{mode}/iter_IoU_{c}', iou)

        self.log(f'{mode}/iter_loss', loss)

        return return_dict

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')

    def __share_epoch_end(self, outputs, mode):

        # loss
        losses = []
        for out in outputs:
            losses.append(out['loss'].cpu().detach().numpy())
        losses = np.mean(losses)
        self.log(f'{mode}/epoch_loss', losses)

        mean_iou = 0
        mean_f1 = 0

        # metrics
        for c in range(4):
            tps, tns, fps, fns, precisions, recalls, f1s, IoUs = \
                [], [], [], [], [], [], [], []
            for out in outputs:
                # assert False, (out[f'TP_{c}'], out[f'TP_{c}'].shape)
                for (tp, tn, fp, fn, precision, recall, f1, iou) in zip(out[f'TP_{c}'],
                                                                        out[f'TN_{c}'],
                                                                        out[f'FP_{c}'],
                                                                        out[f'FN_{c}'],
                                                                        out[f'Precision_{c}'],
                                                                        out[f'Recall_{c}'],
                                                                        out[f'F1_{c}'],
                                                                        out[f'IoU_{c}'],):

                    tps.append(tp.unsqueeze(dim=0))
                    tns.append(tn.unsqueeze(dim=0))
                    fps.append(fp.unsqueeze(dim=0))
                    fns.append(fn.unsqueeze(dim=0))

                    precisions.append(precision.unsqueeze(dim=0))
                    recalls.append(recall.unsqueeze(dim=0))
                    f1s.append(f1.unsqueeze(dim=0))
                    IoUs.append(iou.unsqueeze(dim=0))

            tps = torch.cat(tps, dim=0).squeeze()
            tns = torch.cat(tns, dim=0).squeeze()
            fps = torch.cat(fps, dim=0).squeeze()
            fns = torch.cat(fns, dim=0).squeeze()

            precisions = torch.cat(precisions, dim=0).squeeze()
            recalls = torch.cat(recalls, dim=0).squeeze()
            f1s = torch.cat(f1s, dim=0).squeeze()
            IoUs = torch.cat(IoUs, dim=0).squeeze()

            # logging
            self.log(f'{mode}/epoch_TP_{c}', tps)
            self.log(f'{mode}/epoch_TN_{c}', tns)
            self.log(f'{mode}/epoch_FP_{c}', fps)
            self.log(f'{mode}/epoch_FN_{c}', fns)

            self.log(f'{mode}/epoch_Precision_{c}', precisions)
            self.log(f'{mode}/epoch_Recall_{c}', recalls)

            self.log(f'{mode}/epoch_F1_{c}', f1s)
            self.log(f'{mode}/epoch_IoU_{c}', IoUs)

            mean_iou += np.mean(IoUs.numpy()).item()
            mean_f1 += np.mean(f1s.numpy()).item()

        mean_iou /= 4
        mean_f1 /= 4
        self.log(f'{mode}/mean_IoU', mean_iou)
        self.log(f'{mode}/mean_F1', mean_f1)

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]


if __name__ == '__main__':

    class CFG(object):
        # basic
        debug: bool = False
        debug_sample: int = 32
        folds: int = 5
        seed: int = 417
        eps: float = 1e-12
        outdir: str = '../../train/output/flood/'

        # data
        PATH_FOLD_CSV: str = f'../../data/folds/'
        csv_state: str = 'clean-v3_'  # area_, clean-v1_, clean-v2_ '', fulltrain_,

        # train
        epoch: int = 200
        trainer: Dict[str, Any] = {
            'gpus': 1,
            'accumulate_grad_batches': 1,
            'progress_bar_refresh_rate': 1,
            'stochastic_weight_avg': False,
            'fast_dev_run': False,
            'num_sanity_val_steps': 0,
            'resume_from_checkpoint': None,
            'check_val_every_n_epoch': 2,
            'val_check_interval': 1.0,
            'precision': 16,
            'gradient_clip_val': 20.,
            'gradient_clip_algorithm': "value"
            # 'accelerator': 'ddp',
        }
        optimizer: Dict[str, Any] = {
            'name': 'optim.AdamW',
            'params': {
                'lr': 1e-4,
            },
        }
        scheduler: Dict[str, Any] = {
            'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
            'params': {
                'T_0': 40,
                'eta_min': 1e-5,
            }
        }
        model: Dict[str, Any] = {
            "architecture": 'unet',  # timmu, smp, unet
            "threshold": 0.4,

            # smp loss mode: https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html
            'loss_mode': 'multilabel',  # 'binary', 'multiclass' ,'multilabel'
            'loss': None,

            'in_channels': 0,
            'out_channels': 4,

            # unet++ :https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/decoders/unetplusplus/model.html
            'decoder_channels': [int(256 / 2**i) for i in range(5)],
            'encoder_name': 'efficientnet-b0',
            'act': None,
            'decoder_use_batchnorm': True,  # True, False, 'inplace'
            # 'dropout_rato': 0.1,

            # TimmUNet
            'timmu': {
                'encoder_params': {
                    "encoder": "hrnet_w18",
                    "decoder_filters": [48, 96, 176, 192],
                    "last_upsample": 32,
                }
            },
            'unet_params': {
                # https://smp.readthedocs.io/en/latest/models.html#unet
                'encoder_name': 'resnet34',
                'encoder_depth': 4,
                'encoder_weights': 'imagenet',
                'decoder_use_batchnorm': True,
                'decoder_channels': [int(128 / 2**i) for i in range(4)],
                'decoder_attention_type': None,
                'activation': None,
            },
        }
        model['loss'] = f'MultiBCEDiceLoss(raito=0.5, mode=\'{model["loss_mode"]}\')'
        mixup: float = 0.0

        thick = {
            'use': False,
            'kernel': (3, 3)
        }

        train_loader: Dict[str, Any] = {
            'batch_size': 8,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': False,
            'drop_last': True,
        }
        val_loader: Dict[str, Any] = {
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': False,
            'drop_last': False
        }

        # preprocess
        features: List[str] = ["preimg", "postimg",
                               "building", "road", "roadspeed", "flood"]
        # ["preimg","postimg","building","road","roadspeed","flood"]

        preprocess: Dict = {
            "input_size": 1312,
        }

        # logging
        project: str = "SpaceNet8_flood"
        runname: str = "3090"
        group: str = f'3090_V6_{csv_state}_{model["architecture"]}_IMG{preprocess["input_size"]}_resnet34-dec128_fl-tr_b8'
        notebook: str = 'baseline_flood.py'

        # post info
        augmentation: str = ''
        fold: int = -1

        # channels
        for f in features:
            if f == 'preimg':
                model['in_channels'] += 3
            if f == 'postimg':
                model['in_channels'] += 3

        if debug:
            epoch = 2
            group = 'DEBUG'

    # box
    cfg = Box({k: v for k, v in dict(vars(CFG)).items() if '__' not in k})

    # 乱数のシードを設定
    seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    pprint(cfg)

    # augmentation
    tf_dict = {

        'train': A.Compose(
            [
                # A.CoarseDropout(max_holes=4, max_height=4, max_width=4,
                #                     min_holes=None, min_height=None, min_width=None,
                #                     fill_value=0.15, mask_fill_value=0.0, always_apply=False, p=0.25),
                # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1,
                #                     border_mode=4, value=None, mask_value=None, always_apply=False,
                #                     approximate=False, same_dxdy=False, p=0.25),
                # A.GridDistortion(num_steps=5, distort_limit=0.4, interpolation=1,
                #                     border_mode=4, value=None, mask_value=None, always_apply=False, p=0.25),
                # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=1,
                #                     border_mode=4, value=0.01, mask_value=0.0, shift_limit_x=None, shift_limit_y=None,
                #                     p=0.5),
                # A.OneOf([
                #     # A.GaussNoise(var_limit=(1e-3, 8e-1), mean=0.15, p=0.5),
                #     A.Blur(blur_limit=3, p=0.25),
                #     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.5),
                # ], p=0.75),
                A.Transpose(p=0.25),
                A.Flip(p=0.5),
                # A.HueSaturationValue (hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
                # A.Rotate(limit=30, p=0.5),
                # DualHSV(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
                # PostFog(fog_coef_lower=0.35, fog_coef_upper=0.85, alpha_coef=0.08, p=0.5),
                A.Resize(cfg.preprocess.input_size, cfg.preprocess.input_size,
                         interpolation=cv2.INTER_LINEAR),  # cv2.INTER_LINEAR, cv2.INTER_NEAREST
                # A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                #             std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ),
        'val': A.Compose(
            [
                A.Resize(cfg.preprocess.input_size, cfg.preprocess.input_size,
                         interpolation=cv2.INTER_LINEAR),  # cv2.INTER_LINEAR, cv2.INTER_NEAREST
                # A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                #             std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ),
    }
    cfg.augmentation = tf_dict

    for fold in range(cfg.folds):
        print(f'#'*60)
        print(f'### Fold: {fold}')
        print(f'#'*60)

        # Setting
        cfg.fold = fold
        wandb_logger = WandbLogger(
            config=cfg,
            name=f"{cfg.runname}_fold{fold}",
            project=cfg.project,
            group=cfg.group,
            tags=[f'fold{fold}', '3090', 'notebook', 'flood'],
            # entity='spaceshift',
        )

        # Data
        datamodule = SpaceNnet8Module(fold, cfg)

        # Model
        model = SpaceNet8Model(cfg)

        # PATH
        dirpath = f'{cfg.outdir}{cfg.group}/{cfg.runname}_fold-{fold}/'
        filename = f"best_fold-{fold}"
        best_model_path = dirpath + filename + ".ckpt"

        # Logging
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor="val/mean_IoU",
            save_top_k=2,
            mode="max",
            save_last=True,
        )

        wandb.save(cfg.notebook)

        print(f'### Start Trainig')
        # Train
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=cfg.epoch,
            callbacks=[lr_monitor, loss_checkpoint],
            **cfg.trainer,
        )

        # saving
        os.makedirs(dirpath, exist_ok=True)
        cfg.augmentation = str(tf_dict).replace('\n', '').replace(' ', '')
        with open(f'{dirpath}cfg.json', 'w') as f:
            json.dump(cfg.to_dict(), f, indent=4)

        # 実行
        cfg.augmentation = tf_dict
        trainer.fit(model, datamodule=datamodule)
        wandb.save(cfg.notebook)
        wandb.finish()
        break
