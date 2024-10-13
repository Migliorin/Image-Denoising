import torch
from torchvision.transforms import v2

def get_transform_v1():
    return v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True)
    ])

def get_transform_v2():
    return v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_transform_v3():
    return v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True)
    ])

