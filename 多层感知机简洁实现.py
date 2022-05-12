import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch

num_inputs,num_outputs,num_hiddens = 784,10,256
#构建网络结构
net = nn.Sequential(d2lzh_pytorch.FlattenLayer(),nn.Linear(num_inputs,num_hiddens),nn.ReLU(),nn.Linear(num_hiddens,num_outputs))
#初始化参数
for param in net.parameters():
    init.normal_(param,mean=0,std=0.01)

batch_size = 256
train_iter,test_iter = d2lzh_pytorch.load_data_fashion_mnist(batch_size)#加载训练集和测试集

loss = torch.nn.CrossEntropyLoss()#设置损失函数

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)#设置优化器

num_epochs = 5
#训练网络
d2lzh_pytorch.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)
