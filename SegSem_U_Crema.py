

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DeeplabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeeplabV3Plus, self).__init__()
        
        self.reduce_resolution = nn.Sequential(
                                               nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                               nn.BatchNorm2d(128),
                                               nn.ReLU(inplace=True)
                                               )

        # Encoder
        self.resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.layer1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1) # 64
        self.layer2 = self.resnet.layer2 # 128
        self.layer3 = self.resnet.layer3 # 256
        self.layer4 = self.resnet.layer4 # 512

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.aspp2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(256 * 5, 256, kernel_size=1, stride=1)

        # Decoder
        self.conv3 = nn.Conv2d(256, 48, kernel_size=1, stride=1)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(320, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

    # ASPP
        aspp1 = self.aspp1(x4)
        aspp2 = self.aspp2(x4)
        aspp3 = self.aspp3(x4)
        aspp4 = self.aspp4(x4)
        global_avg_pool = self.global_avg_pool(x4)
        global_avg_pool = self.conv1(global_avg_pool)
        global_avg_pool = F.interpolate(global_avg_pool, size=aspp4.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate
        concat = torch.cat((aspp1, aspp2, aspp3, aspp4, global_avg_pool), dim=1)
        concat = self.conv2(concat)

        # Decoder
        dec0 = self.conv3(concat)
        dec1 = self.up1(dec0)
        dec1 = F.interpolate(dec1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec1, x3), dim=1)
        dec3 = self.conv4(dec2)
        dec4 = self.conv5(dec3)
        dec5 = self.up2(dec4)
        dec5 = F.interpolate(dec5, size=x1.size()[2:], mode='bilinear', align_corners=True) # Fix size mismatch
        dec6 = torch.cat((dec5, x1), dim=1)
        dec7 = self.conv6(dec6)

        return F.interpolate(dec7, size=x.size()[2:], mode='bilinear', align_corners=True)

model = DeeplabV3Plus(num_classes=3)
#♥model.load_state_dict(torch.load(r'C:/Bloodcells/Modele/Data/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB/Learn_0/pytorch3105_model-state_dict.pth'))
#model= torch.load('C:/Bloodcells/Modele/Data/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB/Learn_0/pytorch3105.pth', map_location=torch.device('cpu'))
if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
# Boucle d'entraînement
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import glob
from PIL import Image
import numpy as np
import random





# Example usage:

import torchvision.transforms as transforms
import torch

class RandomRGBMultiplier(object):
    def __call__(self, img):
        r, g, b = img.split()
        r = transforms.functional.adjust_brightness(r, random.uniform(0.5, 1))
        g = transforms.functional.adjust_brightness(g, random.uniform(0.5, 1))
        b = transforms.functional.adjust_brightness(b, random.uniform(0.5, 1))
        return Image.merge("RGB", [r, g, b])
    

# Example usage:
#transform = transforms.Compose([
#    transforms.RandomChoice([
#        transforms.RandomHorizontalFlip(p=0.5),
#        transforms.Compose([
#            RandomRGBMultiplier(),
#            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5)
#        ])
#    ]),
#    transforms.ToTensor(),
#])
from albumentations.pytorch import ToTensorV2
from albumentations import (MotionBlur,MedianBlur,VerticalFlip,CenterCrop,Crop,Compose,Blur,    RandomRotate90,    ElasticTransform,    GridDistortion,    IAASharpen,    RandomSizedCrop,    IAAEmboss,    OneOf,    CLAHE,    RandomBrightnessContrast,        RandomGamma,    GaussNoise,    RandomResizedCrop)

 


    
# Créer le scheduler
#scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

# Paramètres d'entraînement
num_epochs = 50
batch_size = 5
learning_rate = 0.01
momentum=0.95
weight_decay=0.000001
import cv2
# Put the model on the CPU
device = torch.device('cuda')
model.to(device)


# Fonction de perte et optimiseur
#criterion = nn.NLLLoss()
criterion=nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Définir les transformations
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=179, p=1,interpolation = cv2.INTER_NEAREST),
    A.RandomResizedCrop(height=360, width=360, scale=(0.75, 1.3)),
    A.RGBShift(r_shift_limit=70, g_shift_limit=70, b_shift_limit=70, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
    ToTensorV2(p=1),
])

# Chargement des données
image_filenames = glob.glob('C:\Bloodcells\Modele\Data\Raabin\GrTh\Learn\Images_base/*.png')
label_filenames = glob.glob('C:\Bloodcells\Modele\Data\Raabin\GrTh\Learn\Labels_base/*.png')
#label_filenames = [os.path.join('C:/Bloodcells/Modele/Label/', 'label_' + os.path.basename(fn)) for fn in image_filenames]
images = []
masks = []
for i, filename in enumerate(image_filenames):
    image = Image.open(filename)
    mask = Image.open(label_filenames[i])
    transformed_data = transforms(image=np.array(image), mask=np.array(mask))
    transformed_image = transformed_data['image']
    transformed_mask = transformed_data['mask']
    images.append(transformed_image)
    masks.append(transformed_mask)

dataset = [(image, mask) for image, mask in zip(images, masks)]
# Appliquer les transformations aux données d'entraînement



#train_transforms = RandomTransforms()



class TransformLabels:
    def __call__(self, label):
        label = np.array(label)
        label[label == 0] = 0
        label[label == 60] = 1
        label[label == 100] = 2
        label[label == 140] = 3
        label[label == 180] = 4
        label[label == 220] = 5

        return torch.from_numpy(label).long()

dataset = [(data, TransformLabels()(label)) for (data, label) in dataset]

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # Décrémentation du learning rate
    if (epoch+1) % 200 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
        print(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f}")
    
    for i, (images, labels) in enumerate(train_loader):
    # Envoi des données sur le device
        images, labels = images.to(device), torch.squeeze(labels, dim=1).long().to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
    
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Accuracy: {accuracy:.2f}%")
torch.save(model,'C:\Bloodcells\Modele\Data\Raabin\GrTh\Learn\Labels_base/pytorch0906.pth')
#torch.save(model,'C:\Bloodcells\Modele\pytorch2802.pth')
