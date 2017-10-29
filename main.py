#coding:utf8
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import DogCat
from config import DefaultConfig
import shutil
from tensorboard import SummaryWriter
from train_test import *

opts = DefaultConfig()

if opts.pretrained:
    print('using pretrained model {}'.format(opts.model))
    model = models.__dict__[opts.model](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
###############适用于alexnet和vgg16###############################
    new_classifier = nn.Sequential(*list(model.classifier)[:-1])
    new_classifier.add_module('6', nn.Linear(model.classifier[-1].in_features, 2))
    model.classifier = new_classifier
##################################################################    
else:
    print('creating model {}'.format(opts.model))
    model = models.__dict__[opts.model]()
    new_classifier = nn.Sequential(*list(model.classifier)[:-1])
    new_classifier.add_module('6', nn.Linear(model.classifier[-1].in_features, 2))
    model.classifier = new_classifier

if opts.resume:
    checkpoint = torch.load(opts.resume_file)
    opts.start_epoch = checkpoint['epoch']
    best_prec = checkpoint['best_prec']
    model.load_state_dict(checkpoint['state_dict'])
    print('loaded checkpoint epoch {}'.format(checkpoint['epoch']))
else:
    best_prec = 0.0
if opts.use_gpu:
    model = model.cuda()

train_data = DogCat(opts.train_data_root,train=True)
val_data = DogCat(opts.train_data_root,train=False)
train_dataloader = DataLoader(train_data,opts.batch_size,
                    shuffle=True,num_workers=opts.num_workers)
val_dataloader = DataLoader(val_data,opts.batch_size,
                    shuffle=False,num_workers=opts.num_workers)

criterion = torch.nn.CrossEntropyLoss()
if opts.use_gpu:
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if opts.tensorboard:
    writer = SummaryWriter()
else:
    writer = False
for epoch in range(opts.start_epoch, opts.epoch):
    adjust_learning_rate(opts, optimizer, epoch)
    train(opts, train_dataloader, model, criterion, optimizer, epoch, writer)
    prec = validate(opts, val_dataloader, model, criterion, writer) 
    if epoch % opts.save_epoch == 0 and epoch!=0:
        if prec > best_prec:
            best_prec = prec
            best_epoch = epoch
        torch.save({
        'epoch': epoch+1,
        'arch': opts.model,
        'state_dict': model.state_dict(),
        'best_prec': best_prec}, filename='checkpoint_'+epoch+'_.pth')
shutil.copyfile(opts.checkpointfile+'checkpoint_'+best_epoch+'_.pth', opts.checkpointfile+'best_checkpoint_'+epoch+'_.pth')
if opts.tensorboard:
    writer.close()
