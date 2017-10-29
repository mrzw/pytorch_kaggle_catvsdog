#coding:utf8
import torch
from torch.autograd import Variable
import torch.nn as nn
import time

def train(params, train_loader, model, criterion, optimizer, num_epochs, writer):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    running_datatime = 0.0
    running_batchtime = 0.0
    num_images = 0.0
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        running_datatime += time.time() - end;
        if params.use_gpu:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)               
        outputs = model(images)
        _, y_preds = torch.max(outputs.data,1)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
        running_correct += torch.sum(y_preds==labels.data)
        num_images += images.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_batchtime += time.time() - end;
        end = time.time()
        if i % params.print_freq == 0 and i!=0:
            avg_loss = running_loss/params.print_freq
            avg_acc = running_correct/num_images
            avg_datatime = running_datatime/params.print_freq
            avg_batchtime = running_batchtime/params.print_freq
            print('Train Epoch: [{0}][{1}/{2}]\t Data_time: {:.4f} Batch_time: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format
            (num_epochs, i, len(train_loader), avg_datatime, avg_batchtime, avg_loss, avg_acc))
            running_loss = 0.0
            running_correct = 0.0
            running_datatime = 0.0
            running_batchtime = 0.0
            num_images = 0.0
            if writer:
                writer.add_scalar('train/loss', avg_loss, i)
                writer.add_scalar('train/acc', avg_acc, i)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), i)
                
def validate(params, val_loader, model, criterion, writer):
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    running_datatime = 0.0
    running_batchtime = 0.0
    num_images = 0.0
    all_correct = 0.0
    all_images = 0.0
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        running_datatime += time.time() - end;
        if params.use_gpu:
            images = Variable(images.cuda(),volatile=True)
            labels = Variable(labels.cuda(),volatile=True)
        else:
            images = Variable(images,volatile=True)
            labels = Variable(labels,volatile=True)               
        outputs = model(images)
        _, y_preds = torch.max(outputs.data,1)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
        running_correct += torch.sum(y_preds==labels.data)
        all_correct += torch.sum(y_preds==labels.data)
        num_images += images.size(0)
        all_images += images.size(0)
        running_batchtime += time.time() - end;
        end = time.time()  
        if i % params.print_freq == 0 and i!=0:
            avg_loss = running_loss/params.print_freq
            avg_acc = running_correct/num_images
            avg_datatime = running_datatime/params.print_freq
            avg_batchtime = running_batchtime/params.print_freq
            print('Val: [{1}/{2}]\t Data_time: {:.4f} Batch_time: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format
            (i, len(val_loader), avg_datatime, avg_batchtime, avg_loss, avg_acc))
            running_loss = 0.0
            running_correct = 0.0
            running_datatime = 0.0
            running_batchtime = 0.0
            num_images = 0.0
            if writer:
                writer.add_scalar('val/loss', avg_loss, i)
                writer.add_scalar('val/acc', avg_acc, i)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), i)
    acc = all_correct/all_images
    print('All average accuracy {}'.format(acc))
    return acc

def test(params, test_loader, model):
    model.eval()
    results = []
    for i, (images, labels) in enumerate(test_loader):
        if params.use_gpu:
            images = Variable(images.cuda(),volatile=True)
            labels = Variable(labels.cuda(),volatile=True)
        else:
            images = Variable(images,volatile=True)
            labels = Variable(labels,volatile=True)
        outputs = model(images)       
        probability = nn.functional.softmax(outputs)[:,0].data.tolist()
        batch_results = [(path_,probability_) for path_,probability_ in zip(labels,probability)]
        results += batch_results
    write_csv(results, params.result_file)

def adjust_learning_rate(params, optimizer, epoch):
    lr = params.lr * (0.1**(epoch/params.step_size))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)