import torch
from torch.utils import data
from torch import nn
import torchvision.datasets as vis_dataset

from func_import import *
import baseSet

import trainF

def make_dataset():
    pic_rootPath=dataPr.getDataPath()
    pic_transfm=dataPr.imgTransfm(baseSet.picSize)
    # picDataset=dataPr.class_PicDataset(pic_rootPath, pic_transfm)
    # tags=picDataset.getTags()
    picDataset=vis_dataset.ImageFolder(pic_rootPath,pic_transfm)
    tags=picDataset.classes
    divideNum=dataPr.divideDataset(len(picDataset), baseSet.splitRatio)
    # print("DivideNum:",divideNum)
    dataPr.writeTags(tags)
    print(f"TagNum:{len(tags)},Tags:{tags}")
    return tags, data.random_split(picDataset, divideNum)

def train_main(net, train_iter, test_iter, loss, updater, num_epochs, device, baseEpoch):
    for epoch in range(num_epochs):
        print("Epoch:", epoch+baseEpoch)
        train_metrics = trainF.train_epoch(net, train_iter, loss, updater,device)
        if(epoch%3==0):
            print("Train_loss,Train_acc:",train_metrics[0],train_metrics[1])
            val_acc = trainF.val_main(net, test_iter,device)
            print("Val_acc:",val_acc)
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc


def main():
    tags, picDataset= make_dataset()

    train_loader=data.DataLoader(picDataset[0],batch_size=baseSet.batch_size,shuffle=True,num_workers=baseSet.num_workers)
    val_loader=data.DataLoader(picDataset[1],shuffle=True,num_workers=baseSet.num_workers)
    # test_loader=data.DataLoader(picDataset[2],num_workers=baseSet.num_workers)
    
    loss=nn.CrossEntropyLoss()
    model, optimizer, baseEpoch=ResNet.modelLoad(len(tags), baseSet.stateSave, baseSet.device, baseSet.lr)
    train_main(model, train_loader, val_loader, loss,optimizer, baseSet.num_epochs, baseSet.device, baseEpoch)
    dataPr.modelSave(baseSet.num_epochs, model, optimizer, baseSet.stateSave)

if __name__=="__main__":
    main()