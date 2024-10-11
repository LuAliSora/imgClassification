import torch
from torch.utils import data
from torch import nn


from func_import import *
import baseSet


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

def main():
    picDataset=getData.class_PicDataset(resize=baseSet.resize)
    splitRatio=[8,1,1]
    divideNum=getData.divideDataset(len(picDataset),splitRatio)
    train_dataset, val_dataset, test_dataset= data.random_split(picDataset, divideNum)
    train_loader=data.DataLoader(train_dataset,batch_size=baseSet.batch_size,shuffle=True,num_workers=baseSet.num_workers)
    val_loader=data.DataLoader(val_dataset,shuffle=True,num_workers=baseSet.num_workers)
    test_loader=data.DataLoader(test_dataset,num_workers=baseSet.num_workers)
    
    tags=picDataset.getTags()
    # model=ResNet.ResNet_main(input_channels=3,tagNum=len(tags)).to(baseSet.device)
    model=ResNet.ResNet_transL(tagNum=len(tags)).to(baseSet.device)
    
    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=baseSet.lr, amsgrad=True)
    
    train_main(model,train_loader,val_loader,loss,optimizer,baseSet.num_epochs,baseSet.device)
    modelSave(baseSet.num_epochs,model,optimizer)

if __name__=="__main__":
    main()