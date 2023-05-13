import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from models import DnCNN,DnCNN2
from utils import *
import matplotlib.pyplot as plt
# from test import main
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset




data_dir = '/home/nurit/Backup/nurit/DnCNN-PyTorch/data/'
transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
datasetpsnr = datasets.ImageFolder(data_dir, transform=transform)
dataloaderpsnr = torch.utils.data.DataLoader(datasetpsnr, batch_size=32, shuffle=True)

horse_dset = Subset(datasetpsnr, range(0, 12))
ls = DataLoader(horse_dset, batch_size=15, shuffle = True)
test_images = next(iter(ls))[0]
test_images = test_images[:,0:1,:,:]#.cuda()

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def normalize(data):
    return data/255.

net1 = DnCNN(channels=1, num_of_layers=17)
device_ids = [0]
# model1 = nn.DataParallel(net1, device_ids=device_ids)
device = torch.device('cpu')

model1 = nn.DataParallel(net1).cpu()
# model1 = model1#.cuda()
# model1.load_state_dict(torch.load(os.path.join("/home/nurit/Backup/nurit/DnCNN-PyTorch/logs/DnCNN-S-15/", 'net.pth')))
model1.load_state_dict(torch.load("/home/nurit/Backup/nurit/DnCNN-PyTorch/logs/DnCNN-S-15/net.pth", map_location = device))
model1.eval()

net2 = DnCNN2(channels=1, num_of_layers=17)
# model2 = nn.DataParallel(net2, device_ids=device_ids)
model2 = net2.cpu()
# model2 = model2#.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model2.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

iterations = 2000

scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1200,1800], gamma=0.1)
loss_list = []
psnr_list = []

# Img = cv2.imread('/home/nurit/Backup/nurit/DnCNN-PyTorch/data/train/test_390.png')
# Img = cv2.imread('/home/nurit/Backup/nurit/DnCNN-PyTorch/Rona-green-1-700x700.jpg') #sensitivity-to-contrast.jpg')

data_dir = '/home/nurit/Backup/nurit/images/'
transform = transforms.Compose([transforms.RandomCrop(70),
transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# ISource = next(iter(dataloader))[0]
# noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=10.0/255.)
# inputs = ISource + noise
# inputs = inputs[:,0:1,:,:]#.cuda()
model2.train()
inputsmat = torch.zeros(128,1,70,70)

for i in range(128):
    ISource = next(iter(dataloader))[0] 
    noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=15.0/255.)
    inputs = ISource + noise
    inputsmat[i] = inputs[:,0:1,:,:]#.cuda()

inputs = inputsmat
psnr_current = 0
for i in range(iterations):
    # ISource = next(iter(dataloader))[0] 
    # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=15.0/255.)
    # inputs = ISource + noise
    # inputs = inputs[:,0:1,:,:]#.cuda()
    # inputs = inputs.reshape(16,1,140,140)
    # inputs = torch.randn(64,1,140,140)#.cuda()
    optimizer.zero_grad()
    Out_teacher = model1(inputs)
    Out_student = model2(inputs)
    loss = criterion(Out_teacher,Out_student)
    loss.backward()
    optimizer.step()
    scheduler2.step()
    # print(i , loss.item())
    # loss_list.append(loss.item())
    if i%100 == 0:
      INoisy = test_images+torch.FloatTensor(test_images.size()).normal_(mean=0, std=15/255.)#.cuda()
      psnr = batch_PSNR(torch.clamp(INoisy-model2(INoisy), 0., 1.), test_images, 1.)
      print(psnr)
      psnr_list.append(psnr)



np.save('/home/nurit/Backup/nurit/DnCNN-PyTorch/psnr_list.npy',psnr_list)
# np.save('/home/nurit/Backup/nurit/DnCNN-PyTorch/loss_list.npy',loss_list)

# loss_list = np.load('/home/nurit/Backup/nurit/DnCNN-PyTorch/loss_list.npy')
# psnr_list = np.load('/home/nurit/Backup/nurit/DnCNN-PyTorch/psnr_list.npy')



fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('loss', color=color)
ax1.plot(loss_list,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0,0.002])
ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('psnr', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(1,iterations,100),psnr_list, '*', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# plt.plot(loss_list,color=color)
# plt.plot(np.arange(1,iterations,100),psnr_list, '*')
plt.savefig('/home/nurit/Backup/nurit/DnCNN-PyTorch/loss_list.png',bbox_inches='tight')


inputs = torch.randn(1,1,35,35)#.cuda()
model1.eval()
torch.save(model2.state_dict(), '/home/nurit/Backup/nurit/DnCNN-PyTorch/model2.pth')

model2.load_state_dict(torch.load('/home/nurit/Backup/nurit/DnCNN-PyTorch/model2.pth'))
model2.eval()
bestmodel2 = nn.DataParallel(net2, device_ids=device_ids)#.cuda()
bestmodel2.load_state_dict(torch.load('/home/nurit/Backup/nurit/DnCNN-PyTorch/bestmodel2.pth'))
bestmodel2.eval()



f, axarr = plt.subplots(1, 3, figsize = (16,16))
axarr = axarr.flatten()
Out_teacher = model1(inputs)
Out_student = model2(inputs)
axarr[0].imshow((Out_student[0]).permute(1,2,0).detach().cpu().numpy())
axarr[0].set_title("Out_student")
axarr[1].imshow((Out_teacher[0]).permute(1,2,0).detach().cpu().numpy())
axarr[1].set_title("Out_teacher")
im4 = axarr[2].imshow(((Out_student-Out_teacher)[0]).permute(1,2,0).detach().cpu().numpy())
axarr[2].set_title("Out_student-Out_teacher")
# axarr[2].colorbar()
cbar = plt.colorbar(im4)
plt.savefig('/home/nurit/Backup/nurit/DnCNN-PyTorch/Out_student_Out_teacher.png',bbox_inches='tight')

