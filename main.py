'''
Adapted from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
'''

import torch
import torchvision
import torch.nn as nn
import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import PIL.ImageOps
from PIL import Image

class VDataset(Dataset):
    
    def __init__(self,imageDataset,check_criterion,folder=True,transform=None,should_invert=True):
        
        if folder:
            self.imageDataset = imageDataset.imgs
        else:
            self.imageDataset = imageDataset
        
        self.folder = folder
        self.transform = transform
        self.should_invert = should_invert
        self.check_criterion = check_criterion
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageDataset)
        
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 

        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageDataset) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageDataset) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        if self.folder:
            img0 = Image.open(img0_tuple[0])
            img1 = Image.open(img1_tuple[0])

            img0 = img0.convert("L")
            img1 = img1.convert("L")
            
            if self.should_invert:
                img0 = PIL.ImageOps.invert(img0)
                img1 = PIL.ImageOps.invert(img1)
        else:
            img0 = img0_tuple[0]
            img1 = img1_tuple[0]

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(self.check_criterion(img1_tuple[1],img0_tuple[1]))],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageDataset)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VerifierNetwork(nn.Module):
    def __init__(self,ch=1):
        super(VerifierNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 32, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            
            Flatten(),
            nn.Linear(32*9*9, 2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True), 
            
            nn.Linear(2048, 1024)

        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True), 
            
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(16, 1),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)

        output = torch.cat((output1, output2), 1)
        similarity = self.fc(output)

        return similarity, output1, output2

def imshow(img,path,text=None,should_save=False):
    npimg = img.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.axis("off")
    
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

        ax.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)

parser = argparse.ArgumentParser()

parser.add_argument('--training_dir', type=str, default="./data/faces/training/")
parser.add_argument('--testing_dir', type=str, default="./data/faces/testing/")
parser.add_argument('--dataset', type=str, default="faces")
parser.add_argument('--b_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--resume', action='store_true', default=False)

if __name__ == "__main__":

    opt = parser.parse_args()
    
    if opt.dataset == 'faces':
        dataset = dset.ImageFolder(root=opt.training_dir)
        dataset_test = dset.ImageFolder(root=opt.testing_dir)

        transf = [transforms.Resize((50,50)), transforms.ToTensor()]
        itr_show = 10
        epoch_show = 5
        folder = True
        channels = 1

    elif opt.dataset == 'mnist':
        dataset = dset.MNIST('./data/', train=True, download=True,transform=transforms.ToTensor())
        dataset_test = dset.MNIST('./data/', train=False, download=True,transform=transforms.ToTensor())
        
        transf = [transforms.ToPILImage(), transforms.Resize((50,50)), transforms.ToTensor()]
        itr_show = 67 #134
        epoch_show = 10
        folder = False
        channels = 1

    elif opt.dataset == 'cifar-10':
        dataset = dset.CIFAR10('./data/', train=True, download=True,transform=transforms.ToTensor())
        dataset_test = dset.CIFAR10('./data/', train=False, download=True,transform=transforms.ToTensor())
        
        transf = [transforms.ToPILImage(), transforms.Resize((50,50)), transforms.ToTensor()]
        itr_show = 56
        epoch_show = 10
        folder = False
        channels = 3
   
    criterion = nn.BCELoss(reduction='sum').cuda()
    check_criterion = lambda x1,x2: x1 == x2

    v_dataset = VDataset(imageDataset=dataset,
                        folder=folder,
                        check_criterion=check_criterion,
                        transform=transforms.Compose(transf),
                        should_invert=False)

    train_dataloader = DataLoader(v_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=opt.b_size)

    net = VerifierNetwork(channels).cuda()

    if opt.resume:
        state = torch.load("./data/model/net.pth")
        net.load_state_dict(state)

    else:
        optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

        print("Training")
        
        for epoch in range(0,opt.n_epochs):
            for i, data in enumerate(train_dataloader,0):
                
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                optimizer.zero_grad()

                similarity, output1, output2 = net(img0,img1)

                loss = criterion(similarity,label)

                loss.backward()
                
                optimizer.step()

                if i % itr_show == 0 and epoch % epoch_show == 0:
                    print("Epoch number {}\nIteration {}\nCurrent loss {}\n".format(epoch, i,loss.item()))

        torch.save(net.state_dict(), "./data/model/net.pth")

    print("Testing")
    v_dataset = VDataset(imageDataset=dataset_test,
                        folder=folder,
                        check_criterion=check_criterion,
                        transform=transforms.Compose(transf),
                        should_invert=False)

    test_dataloader = DataLoader(v_dataset,num_workers=6,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)

    for i in range(10):
        x0,x1,label2 = next(dataiter)

        x0,x1,label2 = x0.cuda(),x1.cuda(),label2.cuda()
        concatenated = torch.cat((x0,x1),0)

        similarity,output1,output2 = net(x0,x1)

        loss = criterion(similarity,label2)
        
        legend = 'L: {:.2f}'.format(loss.item())
        
        if not similarity is None:
            legend +=' | S: {:.2f}'.format(similarity.item())
        
        imshow(torchvision.utils.make_grid(concatenated),"./data/plot/{}.pdf".format(i), legend)
        