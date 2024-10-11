import torch
import classSet

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] >= 1:
        y_hat = y_hat.argmax(axis=1)
    # y_hat=y_hat.flatten()
    # y_hat=y_hat.reshape(-1)
    # print(y.shape,y_hat.shape)
    cmp = y_hat.type(y.dtype) == y
    # print("cmp,smp.shape:",cmp,cmp.shape)
    return float(cmp.type(y.dtype).sum())

def val_main(net, data_iter,device):  #@save
    """计算在指定数据集上模型的精度"""
    net.eval()  # 将模型设置为评估模式
    metric = classSet.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for i,data in enumerate(data_iter):#
            X,y=data[0].to(device),data[1].to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater,device):  #@save

    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = classSet.Accumulator(3)
    # print("device:",device)
    for i,data in enumerate(train_iter):#
        # 计算梯度并更新参数
        X=data[0].to(device)
        y=data[1].to(device)
        y_hat = net(X)
        # print(X.device,y.device)
        # print("X.shape:",X.shape,"y_hat.shape:",y_hat.shape,"y.shape:",y.shape)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]#turple

