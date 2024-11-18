import torch
from torch.utils import data
from torch import nn
import torchvision.datasets as vis_dataset

import argparse

from func_import import *
import baseSet

import trainF


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=20,
        help="batch_size"
    )
    parser.add_argument(
        "--numWk",
        type=int,
        default=4,
        help="num_workers"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=37
    )
    # print(parser.parse_args())
    return parser.parse_args()


def make_dataset():
    pic_rootPath=dataPr.getDataPath("picFile")
    pic_transfm=dataPr.imgTransfm(baseSet.picSize)
    # picDataset=dataPr.class_PicDataset(pic_rootPath, pic_transfm)
    # tags=picDataset.getTags()
    picDataset=vis_dataset.ImageFolder(pic_rootPath,pic_transfm)
    tags=picDataset.classes
    divideNum=dataPr.divideDataset(len(picDataset), baseSet.splitRatio)
    # print("DivideNum:",divideNum)
    dataPr.writeTags(tags, baseSet.tagSave)
    return tags, data.random_split(picDataset, divideNum)

def train_main(net, train_iter, val_iter, loss, updater, num_epochs, device, baseEpoch):
    for epoch in range(num_epochs):
        train_metrics = trainF.train_epoch(net, train_iter, loss, updater, device)
        if(epoch%10==0 or epoch==num_epochs-1):
            print("Epoch:", epoch+baseEpoch)
            print("Train_loss,Train_acc:",train_metrics[0],train_metrics[1])
            val_acc = trainF.val_main(net, val_iter, device)
            print("Val_acc:",val_acc)
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

def test_main(net, test_iter, device):
    test_acc=trainF.val_main(net, test_iter, device)
    print("Test_acc:",test_acc)

def main():
    args=get_args()
    tags, picDataset= make_dataset()

    train_loader=data.DataLoader(picDataset[0],batch_size=args.batch,shuffle=True,num_workers=args.numWk)
    val_loader=data.DataLoader(picDataset[1],shuffle=True,num_workers=args.numWk)
    test_loader=data.DataLoader(picDataset[2],num_workers=args.numWk)
    
    loss=nn.CrossEntropyLoss()
    model, optimizer, baseEpoch=ResNet.modelLoad(len(tags), baseSet.stateSave, baseSet.device, args.lr)
    train_main(model, train_loader, val_loader, loss, optimizer, args.epoch, baseSet.device, baseEpoch)
    test_main(model, test_loader, baseSet.device)
    dataPr.modelSave(baseEpoch+(args.epoch), model, optimizer, baseSet.stateSave)

if __name__=="__main__":
    main()