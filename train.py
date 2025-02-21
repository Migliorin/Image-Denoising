import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer

import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

from losses import MSELossPatchEinops
from modules import TrainModule
from models import TransCLIPRestoration
from transformation import get_transform
from dataset import FFHQDegradationDataset



if __name__ == '__main__':
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        file.close()

    batch_size = params["dataset"]["batch_size"]
    num_workers = params["dataset"]["num_workers"]
    epochs = params["train"]["epochs"]
    patience = params["train"]["patience"]
    name_exp = params["name_exp"]
    dir_save_logs = params["train"]["dir_save_logs"]
    name_to_save = params["train"]["name_to_save"]
    top_k = params["train"]["top_k"]

    noise_transform = get_transform(params["dataset"]["transformation"]["w_noise"])
    normal_transform = get_transform(params["dataset"]["transformation"]["wo_noise"])
    
    train_dataset = FFHQDegradationDataset(
        ffhq_path=params["dataset"]["ffhq_path"],
        transform=normal_transform,
        noise_transform=noise_transform
    )

    
    custom_dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)

    val_dataset = FFHQDegradationDataset(
        ffhq_path=params["dataset"]["val"],
        transform=normal_transform,
        noise_transform=noise_transform
    )

    
    custom_dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    

    model = TransCLIPRestoration(
        img_size=(batch_size,3,512,512),
        patch_size=params["train"]["model"]["patch_size"],
        token_len=params["train"]["model"]["token_len"],
        embed_dim=params["train"]["model"]["token_len"],
        num_heads=params["train"]["model"]["num_heads"],
        num_layers=params["train"]["model"]["num_layers"]
    )
    model = model.cuda()

    loss_fn = MSELossPatchEinops(patch_size=params["train"]["model"]["patch_size"])
    optimizer = torch.optim.Adam(model.parameters(),lr=params["train"]["lr"])
    
    model = TrainModule(model,loss_fn,optimizer)

    early_stopping = EarlyStopping(
        'val_loss',
        patience=patience
    )

    logger = CSVLogger(
        dir_save_logs,
        name=name_exp
    )
    path_checkpoint = f"{dir_save_logs}/{name_exp}/version_{logger.version}"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        dirpath=path_checkpoint,
        filename=name_to_save,
    )

    trainer = Trainer(
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        max_epochs=epochs,
        logger=logger,
        devices=1,
        accelerator="auto"
    )
    
    os.makedirs(path_checkpoint, exist_ok = True)
    
    with open(f"{path_checkpoint}/hparams.yml","w+") as outfile:
        yaml.dump(params,outfile)

    trainer.fit(
        model,
        custom_dataloader_train,
        custom_dataloader_val
    )
