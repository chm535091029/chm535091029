import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d21

n_train,n_test,num_inputs = 20,100,200
true_w,true_b = torch.ones(num_inputs,1)*0.01,0.05 #真实权重和b

features = torch.randn((n_test+n_train,num_inputs)) #生成样本特征，包括训练集和测试集
labels = torch.matmul(features,true_w) + true_b #生成样本标签
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float) #噪声项
train_features, test_features = features[:n_train,:],features[n_train:,:]
train_labels, test_labels = labels[:n_train],labels[n_train:]


batch_size,num_epochs,lr = 1,100,0.003
net,loss = d21.linreg,d21.squard_loss  #线性回归和均方误差

dataset = torch.utils.data.TensorDataset(train_features,train_labels)#把标签和特征和并为每一个样本
train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)#批量读取数据集

def fit_plot(wd):
    net = nn.Linear(num_inputs,1)
    nn.init.normal_(net.weight,mean=0,std=1)#初始化权重和偏差
    nn.init.normal_(net.bias,mean=0,std=1)
    # 用weight_decay对权重进行参数衰减,为权重单独设一个优化器
    optiimzer_w = torch.optim.SGD(params=[net.weight],lr=lr,weight_decay=wd)
    # 用weight_decay对偏差进行参数衰减，为偏差单独设一个优化器
    optiimzer_b = torch.optim.SGD(params=[net.bias],lr=lr)
    train_ls, test_ls = [], []  # 记录每个周期的损失
    for _ in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y).mean()
            l = l.sum()

            optiimzer_w.zero_grad()#backward前梯度清零
            optiimzer_b.zero_grad()
            l.backward()
            #对两个优化器分别更新
            optiimzer_b.step()
            optiimzer_w.step()

        train_ls.append(loss(net(train_features), train_labels).mean().item())  # 每轮记录损失到数组
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d21.semilogy_d21(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,
                     ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

fit_plot(18)