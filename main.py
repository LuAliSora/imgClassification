import torch
from torch.utils import data
from torch import nn

from func_import import *
import baseSet

def make_dataset():
    pic_rootPath=getData.getDataPath()
    pic_transfm=getData.imgTransfm(baseSet.picSize)
    picDataset=getData.class_PicDataset(pic_rootPath, pic_transfm)
    tags=picDataset.getTags()
    divideNum=getData.divideDataset(len(picDataset), baseSet.splitRatio)
    return tags,data.random_split(picDataset, divideNum)

def train_main(net, train_iter, test_iter, loss, updater, num_epochs, device):
    for epoch in range(num_epochs):
        print("Epoch:",epoch)
        train_metrics = trainF.train_epoch(net, train_iter, loss, updater,device)
        if(epoch%3==0):
            print("Train_loss,Train_acc:",train_metrics[0],train_metrics[1])
            test_acc = trainF.val_main(net, test_iter,device)
            print("Test_acc:",test_acc)
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

def modelSave(epochs,model,optimizer,fileSave="modelSave.pth"):
    torch.save({
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        fileSave)
    # print(model.state_dict())
    print("model_save!")

def main():
    tags,picDataset= make_dataset()
    train_loader=data.DataLoader(picDataset[0],batch_size=baseSet.batch_size,shuffle=True,num_workers=baseSet.num_workers)
    val_loader=data.DataLoader(picDataset[1],shuffle=True,num_workers=baseSet.num_workers)
    test_loader=data.DataLoader(picDataset[2],num_workers=baseSet.num_workers)
    # model=ResNet.ResNet_main(input_channels=3,tagNum=len(tags)).to(baseSet.device)
    model=ResNet.ResNet_transL(tagNum=len(tags)).to(baseSet.device)
    
    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=baseSet.lr, amsgrad=True)
    
    train_main(model,train_loader,val_loader,loss,optimizer,baseSet.num_epochs,baseSet.device)
    modelSave(baseSet.num_epochs,model,optimizer)

if __name__=="__main__":
    main()