import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import random
import argparse
import json
import os

from networks import ConvMixer, ConvMixerXL

###========================================================================
#Setup seeds
seed = 1204
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

###========================================================================
#Setup args
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

parser.add_argument('--model', default='CM', choices=['CM','CM-XL'])
parser.add_argument('--skip_period', default=3, help='Denominator in extra skip connection periodicity computation; only used in ConvMixer-XL')
parser.add_argument('--activation', default='GELU', choices=['GELU','ReLU','SiLU'])
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.2, type=float)
parser.add_argument('--ra-m', default=12, type=int)
parser.add_argument('--ra-n', default=2, type=int)
parser.add_argument('--jitter', default=0.2, type=float)
parser.add_argument('--no_aug',action='store_true',help="Enable flag to remove augmentations")

parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=8, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr-max', default=0.005, type=float)
parser.add_argument('--workers', default=8, type=int)

parser.add_argument('--save_dir',default='./',help='Directory to save outputs to')

args = parser.parse_args()

#Check dir exist; if not, create
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
dst = os.path.join(os.getcwd(),args.save_dir)

#Save args
with open(os.path.join(dst,'args_{}.txt'.format(args.name)), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


###========================================================================
#Setup dataset
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

#Transforms
if not args.no_aug:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=args.reprob)
    ])
else:
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

#Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
testvalset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=test_transform)

#Split test-val set
ln = len(testvalset)
testset = testvalset[:ln//2]
valset = testvalset[ln//2:]

#Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)


###========================================================================
#Setup model, optim and scheduler
if args.model == 'CM':
    model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=10, activation=args.activation)
elif args.model == 'CM-XL':
    model = ConvMixerXL(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=10, skip_period=args.skip_period, activation=args.activation)

model = nn.DataParallel(model).cuda()

lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
                                  [0, args.lr_max, args.lr_max/20.0, 0])[0]

opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd) #optimizer
criterion = nn.CrossEntropyLoss() #loss function
scaler = torch.cuda.amp.GradScaler() #grad scaler


###========================================================================
#Training and validation

#Setup vars
train_loss_ls = []
train_acc_ls = []

val_loss_ls = []
val_acc_ls = []


#Train loop
for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0

    #Go through training steps
    for i, (X, y) in enumerate(trainloader):
        
        #Set train mode and port sample to cuda
        model.train()
        X, y = X.cuda(), y.cuda()

        #Step lr scheduler and zero grad
        lr = lr_schedule(epoch + (i + 1)/len(trainloader))
        opt.param_groups[0].update(lr=lr)
        opt.zero_grad()

        #FP and compute loss with amp
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        #Scale gradient and clip norm
        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        #Compute and log loss and acc
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        
    #Go through eval steps
    model.eval() #to evaluation mode first
    val_acc, val_loss, m = 0, 0, 0

    with torch.no_grad(): #no grad needed
        for i, (X, y) in enumerate(valloader):
            X, y = X.cuda(), y.cuda() #port to cuda

            #FP and compute result and log
            with torch.cuda.amp.autocast():
                output = model(X)
            loss = criterion(output,y)

            val_loss += loss.item() * y.size(0)
            val_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    #Log
    train_loss_ls.append({'Epoch':epoch, 'Value': round(train_loss/n,5)})
    val_loss_ls.append({'Epoch':epoch, 'Value': round(val_loss/m,5)})
    train_acc_ls.append({'Epoch':epoch, 'Value': round(train_acc/n,5)})
    val_acc_ls.append({'Epoch':epoch, 'Value': round(val_acc/m,5)})

    print(f'[{args.name}-{args.model}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {val_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')


###========================================================================
#Final test
#Go through with final testing
print("==="*25)
print("[Training Complete. Evaluation with testset initiated.]")


#To evaluation mode first
model.eval()
test_acc, test_loss, m = 0, 0, 0

with torch.no_grad(): #no grad needed
    for i, (X, y) in enumerate(testloader):
        X, y = X.cuda(), y.cuda() #port to cuda

        #FP and compute result and log
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output,y)

        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        m += y.size(0)

    #Log
    test_loss_ls = [{'Value':round(test_loss/m,5)}]
    test_acc_ls = [{'Value':round(test_acc/m,5)}]
    

print(f'[{args.name}-{args.model}] | Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}')


#Save everything
with open(os.path.join(dst,'train_loss_{}.txt'.format(args.name)), 'w') as f:
    json.dump(train_loss_ls, f, indent=4)

with open(os.path.join(dst,'train_acc_{}.txt'.format(args.name)), 'w') as f:
    json.dump(train_acc_ls, f, indent=4)

with open(os.path.join(dst,'val_loss_{}.txt'.format(args.name)), 'w') as f:
    json.dump(val_loss_ls, f, indent=4)

with open(os.path.join(dst,'val_acc_{}.txt'.format(args.name)), 'w') as f:
    json.dump(val_acc_ls, f, indent=4)

with open(os.path.join(dst,'test_loss_{}.txt'.format(args.name)), 'w') as f:
    json.dump(test_loss_ls, f, indent=4)

with open(os.path.join(dst,'test_acc_{}.txt'.format(args.name)), 'w') as f:
    json.dump(test_acc_ls, f, indent=4)

#Save model
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, os.path.join(dst,'{}.pkl'.format(args.name)))