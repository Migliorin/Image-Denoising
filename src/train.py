import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

from models.transformer_torch import VisionModel
from noises import add_noise
from dataset import CustomImageDataset


import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.light_module import LightningVisionTransformer
from torchvision.transforms import v2

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer

class AddNoise(torch.nn.Module):
    def forward(self, img, noise,**kwargs):
        noisy_image = add_noise(img, noise_type=noise,**kwargs)

        return Image.fromarray(noisy_image)

if __name__ == '__main__':
    with open('params.yml', 'r') as file:
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

    noise = AddNoise()
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    path = params["dataset"]["dataframe"]

    df = pd.read_csv(path)

    train = df[df["split"] == 'train']
    test = df[df["split"] == 'test']
    val = df[df["split"] == 'val']

    train_dataset = CustomImageDataset(train,transform,noise)
    test_dataset = CustomImageDataset(test,transform,noise)
    val_dataset = CustomImageDataset(val,transform,noise)

    
    custom_dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)
    
    custom_dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)
    
    custom_dataloader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)
    
    model = VisionModel(
        img_size=(batch_size,3,224,224),
        patch_size=14, 
        token_len=512,
        num_layers=12,
        num_heads=16
    )
    
    model = model.cuda()
    loss_fn = eval(params["train"]["loss"]["model"])
    optimizer = eval(params["train"]["optim"]["model"])
    
    model_ = LightningVisionTransformer(model,loss_fn,optimizer)

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
        devices=2,
        accelerator="auto"
    )
    
    os.makedirs(path_checkpoint, exist_ok = True)
    
    with open(f"{path_checkpoint}/hparams.yml","w+") as outfile:
        yaml.dump(params,outfile)

    trainer.fit(
        model_,
        custom_dataloader_train,
        custom_dataloader_val
    )
