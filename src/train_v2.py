#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


import pandas as pd
from PIL import Image
from tqdm import tqdm

from model_v2 import VisionModel
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
        noisy_image = add_noise(img, noise_type=noise,salt_pepper_amount=0.02,**kwargs)

        return Image.fromarray(noisy_image)

if __name__ == '__main__':
    batch_size = 3
    num_workers = 8
    lr = 0.001
    epochs = 64
    patience = 5
    #dir_path_checkpoint = "/home/lucas/experimentos/"
    name_exp = "Novo_modelo"
    dir_save_logs = "/home/lucas/experimentos/"
    #prefix = "mse_head_14"
    name_to_save = "checkpoint-{epoch:02d}-{val_loss:.2f}"
    top_k = 1

    noise = AddNoise()
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    #path = "/home/lucas/datasets/dataframe_v1.csv"
    path = "/home/lucas/Image-Denoising/dataframes/dataframe_v1.csv"
    df = pd.read_csv(path)

    train = df[df["split"] == 'train']
    val = df[df["split"] == 'val']

    train_dataset = CustomImageDataset(train,transform,noise)
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
    
    model = VisionModel(
        img_size=(batch_size,3,224,224),
        patch_size=14,
        #patch_size=28,
        token_len=512,
        #token_len=1024,
        embed_dim=512,
        num_layers=12,
        num_heads=16
    )
    
    model = model.cuda()
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    model_ = LightningVisionTransformer(model,loss_fn,optimizer)

    early_stopping = EarlyStopping(
        'val_loss',
        patience=patience
    )

    logger = CSVLogger(
        dir_save_logs,
        name=name_exp
        #prefix=prefix
    )
    path_checkpoint = f"{dir_save_logs}/{name_exp}/version_{logger.version}"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="val_loss",
        mode="min",
        #dirpath=dir_path_checkpoint,
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
        accelerator="gpu",
        #strategy="ddp"
    )
    #print(path_checkpoint)

    trainer.fit(
        model_,
        custom_dataloader_train,
        custom_dataloader_val
    )
