import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d21

batch_size = 256
train_iter, test_iter = d21.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs=784,10
# class LinearNet(nn.Module):
#     def __init__(self,num_inputs,num_outputs):
#         super(LinearNet,self).__init__()
#         self.linear = nn.Linear(num_inputs,num_outputs)
#     def forward(self,x):  #x shape:(batch,28,28)
#         y = self.linear(x.view(x.shape[0],-1))
#         return y
# net = LinearNet(num_inputs,num_outputs)

from collections import OrderedDict
#构建网络的结构，将转换层和线性模型依次加入网络层
net = nn.Sequential(OrderedDict([('flatten',d21.FlattenLayer()),('linear',nn.Linear(num_inputs,num_outputs))]))
init.normal_(net.linear.weight,mean=0,std=0.01)#初始化权重参数为正态分布
init.constant_(net.linear.bias,val=0)#初始化偏差为常量0
loss = nn.CrossEntropyLoss()#包括了softmax运算和交叉熵损失计算的函数
optimizer = torch.optim.SGD(net.parameters(),lr=0.1) #使用小批量梯度下降

num_epochs = 5
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)
