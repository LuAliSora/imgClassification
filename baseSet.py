import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
picSize=(640,640)
splitRatio=[8,2]#train, val
stateSave="modelSave.pth"
lr, num_epochs, batch_size, num_workers = 0.05, 37, 20, 4

