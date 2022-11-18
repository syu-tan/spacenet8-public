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
import gc

from tqdm import tqdm
import numpy as np
import pandas as pd
from box import Box
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

import tifffile
from osgeo import gdal
from osgeo import osr

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import jaccard_score
from scipy.optimize import minimize
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from adabelief_pytorch import AdaBelief
import segmentation_models_pytorch as smp
import ttach as tta

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import LightningDataModule, LightningModule

from utils.utils import write_geotiff, TimmUnet

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
pd.options.display.max_colwidth = 250
pd.options.display.max_rows = 30

PATH_CFG = '../../train/output/flood/3090_V5_clean-v3__timmu_IMG1312_hrnet_w18_fl-tr_b4/3090_fold-0/cfg.json'
with open(PATH_CFG) as f:
    cfg = json.load(f)

# box
cfg = Box(cfg)
pprint(cfg)

# UPDATE
# cfg['preprocess'] = {"input_size": 512}
# cfg.model.threshold = 0.45
cfg['val_loader'] = {'batch_size': 1,
                'drop_last': False,
                'num_workers': 1,
                'pin_memory': True,
                'shuffle': False}

# debug
# cfg['group'] = 'DEBUG'
# cfg['debug'] = False

# 1 fold
HANSFOLD = False


# 乱数のシードを設定
seed_everything(cfg.seed)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

pprint(cfg)

# augmentation
tf_dict = {
    'val': A.Compose(
        [
            A.Resize(cfg.preprocess.input_size, cfg.preprocess.input_size,
                    interpolation=1),
            ToTensorV2(),
        ]
    ),
} 

tf_dict['test'] = tf_dict['val']

class SpaceNnet8Dataset(Dataset):
    def __init__(self,
                 fold: int,
                 phase: str,
                 ):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images
        
        Parameters:
        ------------
        fold: (int) 
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv. 
        
        """
        
        self.phase = phase
        
        if phase == 'test':
            self.all_data_types = ["preimg", "postimg"]
            # csv_filename = os.path.join('../../data/Louisiana-West_Test_Public/', 'test_preimg-postimg.csv')
            csv_filename = os.path.join('./test_preimg-postimg.csv')
        else:
            self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
            csv_filename = os.path.join(cfg.PATH_FOLD_CSV, f'fold{fold}_seed{cfg.seed}_{self.phase}.csv')
        
        self.data_to_load = self.all_data_types
        
        self.transform = tf_dict[self.phase]
        
        self.files = []

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None
        
        with open(csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for k, row in enumerate(reader):
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    in_data[j]=row[j]
                self.files.append(in_data)
                
                if cfg.debug and k > cfg.debug_sample:
                    break
        
        print("loaded", len(self.files), "image filepaths")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_dict = self.files[index]

        imgs, masks = [], []
        
        # gather
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath is not None:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                if i == "postimg":
                    ds = self.get_warped_ds(data_dict["postimg"])
                else:
                    ds = gdal.Open(filepath)
                image = ds.ReadAsArray()
                ds = None
            
                if i in ['preimg' ,'postimg']:
                    imgs.append(image.transpose(1, 2, 0))
                else:
                    # 1 channel
                    if len(image.shape) <= 2:
                        masks.append(image[:,:, np.newaxis])
                    else:
                        masks.append(image.transpose(1, 2, 0))
        
        # align channel last
        imgs = np.concatenate(imgs, axis=2)
        
        if self.phase != 'test':
            masks = np.concatenate(masks, axis=2)
        
        # augmentation
        if self.phase == 'test':
            transformed = self.transform(image=imgs)
        else:
            transformed = self.transform(image=imgs, mask=masks)
        
        imgs = transformed["image"]
        
        if self.phase != 'test':
            masks = transformed["mask"].permute(2, 0, 1) # torch channel fast
        
        return imgs, masks

    def get_image_filename(self, index: int) -> str:
        """ return pre-event image absolute filepath at index """
        data_dict = self.files[index]
        return data_dict["preimg"]

    def get_warped_ds(self, post_image_filename: str) -> gdal.Dataset:
        """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks 
        
        SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
        In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
        the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
        ds = gdal.Warp("", post_image_filename,
                       format='MEM', width=1300, height=1300,
                       resampleAlg=gdal.GRIORA_Bilinear,
                       outputType=gdal.GDT_Byte)
        return ds
    
class SpaceNet8Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        # self._criterion = eval(cfg.model.loss)
        
    def __build_model(self):
        if self.cfg.model.architecture == 'smp':
            self.backbone = smp.UnetPlusPlus(encoder_name=cfg.model.encoder_name,
                                                encoder_weights="imagenet",
                                        decoder_attention_type='scse',
                                        in_channels=cfg.model.in_channels, activation=cfg.model.act,
                                        decoder_channels=cfg.model.decoder_channels,
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
            raise ValueError(f'Check `cfg.model.architecture` >>> {self.model.architecture}')

    def forward(self, x):
        feat = self.backbone(x)
        return feat
    
def make_prediction_flood_png(image, postimage, gt, prediction, save_figure_filename):
    raw_im = image.transpose(1, 2, 0) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    post_im = postimage.transpose(1, 2, 0)
    post_im = post_im/np.max(post_im)
    
    combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow'])

    grid = [[raw_im, gt, prediction],[post_im, 0, 0]]

    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    for row in range(2):
        for col in range(3):
            ax = axs[row][col]
            ax.axis('off')
            if row==0 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==1 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==0 and col in [1,2]:
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5], combined_mask_cmap.N))
            elif row==1 and col == 1:
                ax.imshow(grid[0][0])
                mask = np.where(gt==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            elif row==1 and col == 2:
                ax.imshow(grid[0][0])
                mask = np.where(prediction==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename, dpi=95)
    if cfg.debug:
        plt.show();
    plt.clf()
    plt.close(fig)
    plt.close('all')


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'{device=}')
fold = cfg.folds -1

print(f'#'*60)
print(f'### Fold: {fold} HANSFOLD {HANSFOLD}')
print(f'#'*60)
print(f'### Start Inference')

ds_val = SpaceNnet8Dataset(fold, phase='test')
dl_val = DataLoader(ds_val, **cfg.val_loader)

for i, (images, _) in enumerate(dl_val):
    current_image_filename = ds_val.get_image_filename(i)
    print("evaluating: ", f'{i}/{len(ds_val)}', os.path.basename(current_image_filename))
    # print(images.shape, labels.shape)
    
    # cuda
    images = images.to(device)
    images = images.float()
    outputs = torch.zeros((1, 4, cfg.preprocess.input_size,cfg.preprocess.input_size)).to(device)
    
    for fold in range(cfg.folds):
        
        # fold setting
        dirpath = f'{cfg.outdir}{cfg.group}/{cfg.runname}_fold-{fold}/'
        # filename = f"best_fold-{fold}"
        filename = f"last"
        PATH_BEST_MODEL = f'{dirpath}{filename}.ckpt'

        save_preds_dir = f'{dirpath}test_fld/'
        save_fig_dir = f'{dirpath}test_png/'
        os.makedirs(save_preds_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        
        # model
        model = SpaceNet8Model.load_from_checkpoint(PATH_BEST_MODEL, cfg=cfg)
        model.eval()
        if cfg.debug:
            device = torch.device('cpu')

        model = model.to(device)
        # TTA
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

        with torch.no_grad():
            outputs += model(images)
            
            if HANSFOLD:
                break
    
    # mean average and logits
    if HANSFOLD:
        outputs = torch.sigmoid(outputs)
    else:
        outputs = torch.sigmoid(outputs/ cfg.folds) 
            
    outputs = outputs.cpu().numpy()
    images = images.cpu().numpy()
    
    # allocate
    preimg, postimg = images[0, :3, :, :], images[0, 3:, :, :]
    pred_flood = outputs[0, :4, :, :]
    
    # mask
    premask = 1. - (np.mean(preimg, 0) == np.ones((cfg.preprocess.input_size,cfg.preprocess.input_size))).astype(np.float32)
    postmask = 1. - (np.mean(postimg, 0) == np.ones((cfg.preprocess.input_size,cfg.preprocess.input_size))).astype(np.float32)
    mask = premask * postmask
    _pred_flood = pred_flood * np.stack([mask]*4)
    
    # variable threshold
    for vt_w in range(4):
        pred_flood = np.where(_pred_flood > cfg.model.threshold - (0.05 * vt_w), _pred_flood, 0.)
        # casting and background
        pred_flood = np.concatenate([np.zeros((1, *pred_flood.shape[1:])) + 0.01, pred_flood], axis=0) # background+
        pred_flood = np.argmax(pred_flood, axis=0) # 5 channnel -> 1 channel
        
        if np.count_nonzero(pred_flood) > 0:
            # 全面推論抑制
            if np.count_nonzero(pred_flood) > pred_flood.size * 0.75:
                pred_flood = np.zeros(pred_flood.shape, dtype=np.uint8)
                break
            else:
                # 最適閾値
                break 
    
    # resize
    ds = gdal.Open(current_image_filename)
    _pre_img = ds.ReadAsArray()
    # print(_pre_img.shape)
    ds = None
    # _pre_img = tifffile.imread(current_image_filename)
    _, _w, _h = _pre_img.shape

    pred_flood = cv2.resize(
        pred_flood[..., np.newaxis].astype(np.float32), 
        dsize=(_h, _w),
        interpolation=cv2.INTER_NEAREST
        )
    
    ### save prediction
    if save_preds_dir is not None:
        ds = gdal.Open(current_image_filename)
        geotran = ds.GetGeoTransform()
        xmin, xres, rowrot, ymax, colrot, yres = geotran
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjectionRef())
        ds = None
        output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_floodpred.tif")))
        nrows, ncols = pred_flood.shape
        write_geotiff(output_tif, ncols, nrows,
                    xmin, xres, ymax, yres,
                    raster_srs, [pred_flood])
    

    flood_dummy = np.zeros(pred_flood.shape)
    
    if save_fig_dir != None:
        save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
        make_prediction_flood_png(preimg, postimg, flood_dummy, pred_flood[1:], save_figure_filename)
        
    torch.cuda.empty_cache()
    gc.collect()
        
    # early exit
    # if cfg.debug and i > 2: