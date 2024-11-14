import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
picSize=(640,640)
splitRatio=[8,2]#train, val
stateSave="modelSave.pth"
tagSave="tagList.txt"
