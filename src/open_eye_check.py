from PIL import Image 
import torch
import torchvision
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda")

def openEyeCheck(inpIm, model=None):
    def load_sample(file):
        image = Image.open(file)
        image.load()
        return image
    if model is None:
        model = models.mobilenet_v2()
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False), 
            nn.Linear(in_features=1280, out_features=100, bias=True), 
            nn.ReLU(), 
            nn.Linear(in_features=100, out_features=1, bias=True)
        )
        model = model.to(DEVICE)
        model.load_state_dict(torch.load('../weights/weights_mn_50.pth'))
        model.eval()
    
    image = torch.FloatTensor(np.array(load_sample(inpIm))).to(DEVICE).view(1,1,24,24) / 255
    out = torch.sigmoid(model(image)) > 0.5
    
    return int(out.item())

