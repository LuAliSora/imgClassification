import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resize=(800,800)
lr, num_epochs, batch_size, num_workers = 0.05, 10, 40, 4

