import torch
from torch.utils import data
from torch import nn
import torchvision.datasets as vis_dataset

from func_import import *
import baseSet

def make_dataset():
    pic_rootPath=getData.getDataPath()
    pic_transfm=getData.imgTransfm(baseSet.picSize)
    # picDataset=getData.class_PicDataset(pic_rootPath, pic_transfm)
    # tags=picDataset.getTags()
    picDataset=vis_dataset.ImageFolder(pic_rootPath,pic_transfm)
    tags=picDataset.classes
    divideNum=getData.divideDataset(len(picDataset), baseSet.splitRatio)
    # print("DivideNum:",divideNum)
    return tags, data.random_split(picDataset, divideNum)

def train_main(net, train_iter, test_iter, loss, updater, num_epochs, device):
    for epoch in range(num_epochs):
        print("Epoch:",epoch)
        train_metrics = trainF.train_epoch(net, train_iter, loss, updater,device)
        if(epoch%3==0):
            print("Train_loss,Train_acc:",train_metrics[0],train_metrics[1])
            val_acc = trainF.val_main(net, test_iter,device)
            print("Val_acc:",val_acc)
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

def modelLoad(tagNum, fileSave=baseSet.stateSave):
    # model=ResNet.ResNet_main(input_channels=3,tagNum=tagNum).to(baseSet.device)
    model=ResNet.ResNet_transL(tagNum=tagNum).to(baseSet.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=baseSet.lr, amsgrad=True)
    if Path(fileSave).is_file():
        saveState=torch.load(fileSave, weights_only=True)
        model.load_state_dict(saveState['model_state'])
        optimizer.load_state_dict(saveState['optim_state'])
        print("model_load!")
    return model, optimizer

def modelSave(epochs, model, optimizer, fileSave=baseSet.stateSave):
    torch.save({
        'epoch': epochs,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    },
        fileSave)
    # print(model.state_dict())
    print("model_save!")

def main():
    tags, picDataset= make_dataset()
    print(f"TagNum:{len(tags)},Tags:{tags}")
    train_loader=data.DataLoader(picDataset[0],batch_size=baseSet.batch_size,shuffle=True,num_workers=baseSet.num_workers)
    val_loader=data.DataLoader(picDataset[1],shuffle=True,num_workers=baseSet.num_workers)
    # test_loader=data.DataLoader(picDataset[2],num_workers=baseSet.num_workers)
    
    loss=nn.CrossEntropyLoss()
    model, optimizer=modelLoad(len(tags))

    train_main(model, train_loader, val_loader, loss,optimizer, baseSet.num_epochs, baseSet.device)
    modelSave(baseSet.num_epochs, model,optimizer)

if __name__=="__main__":
    main()