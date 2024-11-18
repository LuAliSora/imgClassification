import torch
from torch.utils import data
from pathlib import Path

import torchvision.transforms as vis_trans

from PIL import Image



def getDataPath(folder):
    tempDir=Path.cwd()
    while 1:
        if tempDir.name=='Python0':
            break
        if tempDir==tempDir.parent:
            raise Exception("root_path: Lose!")
        tempDir=tempDir.parent
    picFiles='crawler_firstTry/'+folder
    tempDir=tempDir/picFiles
    # tempDir=tempDir.joinpath(picAll)
    if tempDir.is_dir()==False:
        raise Exception("picFiles_path: Lose!")
    print("GetDataPath:",tempDir)
    return tempDir

def imgTransfm(picSize):
    trans = [vis_trans.Resize(picSize),
            #  vis_trans.RandomCrop(picSize,pad_if_needed=True),
             vis_trans.RandomHorizontalFlip(),
             vis_trans.RandomVerticalFlip(),
             vis_trans.ToTensor()]
    trans = vis_trans.Compose(trans)
    return trans

def getTagPicPath(tags):
    imgs=[]
    labels=[]
    for i,tag in enumerate(tags):
        tagImgs =[str(imgPath) for imgPath in tag.glob('*.jpg')]
        tagName=tag.name
        tagPicSum=len(tagImgs)
        imgs+=tagImgs
        labels+=[i]*tagPicSum
        # print(type(i))
        print(i,tagName,len(tagImgs)) 
    # print(imgs,labels)
    return imgs, labels


#通过创建data.Dataset子类Mydataset来创建输入
class class_PicDataset(data.Dataset):
# 类初始化
    def __init__(self, rootPath, transform):
        self.tags=[tagPath for tagPath in rootPath.iterdir() if tagPath.is_dir()]
        # print("TagPaths:",self.tags)
        self.imgPaths,self.labels=getTagPicPath(self.tags)
        #image transform
        self.transfm=transform
# 进行切片
    def __getitem__(self, index):
        img=Image.open(self.imgPaths[index])
        data=self.transfm(img)
        return data, self.labels[index]
    
# 返回长度
    def __len__(self):
        return len(self.imgPaths)
    
    def getTags(self):
        return [tag.name for tag in self.tags]
    

def divideDataset(total, splitRatio:list):
    divideNum=[int(0.1*r*total) for r in splitRatio]
    divideNum[-1]=total-sum(divideNum[:-1])
    return divideNum
    

def modelSave(epochs, model, optimizer, fileSave):
    torch.save({
        'epoch': epochs,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    },
        fileSave)
    # print(model.state_dict())
    print("Model_Save!")


def writeTags(tags, savePath):
    print("TagNum:",len(tags))
    with open(savePath, "w",encoding="utf-8") as f_tags:
        [f_tags.write(f"{i}\t{tag}\n") for i, tag in enumerate(tags)]

