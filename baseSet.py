import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
picSize=(640,640)
splitRatio=[8,1,1]#train, val, test
stateSave="modelSave.pth"
tagSave="tagList.txt"
