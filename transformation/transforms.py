import torch
from torchvision.transforms import v2

def get_transform(name:str) -> v2.Compose:
    if(name == 'noise'):
        return v2.Compose([
            v2.ToDtype(dtype=torch.float32,scale=False),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif(name == 'normal'):
        return v2.Compose([
            v2.ToDtype(dtype=torch.float32,scale=False)
        ])
    else:
        raise Exception(f"Transformation name not found: {name}")

