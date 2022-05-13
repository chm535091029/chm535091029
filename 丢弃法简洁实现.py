import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d21

#初始化，两个隐层，每层权值用mean=0,std=0.01的正态分布初始化,两个隐藏层的输出都是256
drop_prob1,drop_prob2 = 0.2,0.5 #设置两个隐藏层的丢弃率，一般靠近输入层的要小一点
num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 28*28, 10, 256,256
num_epochs,lr,batch_size = 10,100.0,256
train_iter, test_iter =d21.load_data_fashion_mnist(batch_size) #批量读取数据集到训练集和测试集

net = nn.Sequential(d21.FlattenLayer(),nn.Linear(num_inputs,num_hiddens1),nn.ReLU(),nn.Dropout(drop_prob1),
                    nn.Linear(num_hiddens1,num_hiddens2),nn.ReLU(),nn.Dropout(drop_prob2),nn.Linear(num_hiddens2,num_outputs))

for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss = torch.nn.CrossEntropyLoss() #用于分类任务的交叉熵损失函数
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)
