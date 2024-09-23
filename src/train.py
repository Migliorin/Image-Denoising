from tqdm import tqdm

import pandas as pd

from model import VisionModel
from noises import add_noise
from dataset import CustomImageDataset

from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2




class AddNoise(torch.nn.Module):
    def forward(self, img, noise,**kwargs):
        noisy_image = add_noise(img, noise_type=noise,**kwargs)

        return Image.fromarray(noisy_image)

if __name__ == '__main__':
    batch_size = 4
    num_workers = 8
    lr = 0.001
    epochs = 64

    noise = AddNoise()
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    path = "/home/lucas/datasets/dataframe_v1.csv"
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

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    model = model.train()
    best_test_loss = float('inf')  # Inicializa com um valor muito alto
    
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0
        count_idx = 0
        for ori_img, noi_img in (pbar := tqdm(custom_dataloader_train)):
            noi_img = noi_img.cuda()
            denoised_img = model(noi_img)
            denoised_img = denoised_img.cpu()
            
            loss = loss_fn(denoised_img, ori_img)
    
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            count_idx += 1
            total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss/count_idx}")
    
        test_total_loss = 0
        test_count_idx = 0
        with torch.no_grad():
            for ori_img, noi_img in tqdm(custom_dataloader_val):
                noi_img = noi_img.cuda()
                denoised_img = model(noi_img)
                denoised_img = denoised_img.cpu()
                
                loss = loss_fn(denoised_img, ori_img)
                test_count_idx += 1
                test_total_loss += loss.item()
            
            test_loss_avg = test_total_loss / test_count_idx
            print(f"Test loss: {test_loss_avg}\n")
            
            # Verifica se o test_loss é o menor até agora e salva o modelo
            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Modelo salvo com test_loss: {best_test_loss}\n")
    
