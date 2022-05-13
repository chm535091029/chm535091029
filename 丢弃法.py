import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d21
def dropout(X,drop_prob):#丢弃概率p
    X = X.float()
    assert 0<=drop_prob<=1 #检查是否符合条件[0,1]否则中断直行
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape)<keep_prob).float() #返回0或1元素的矩阵

    return mask*X/keep_prob
X = torch.arange(16).view(2,8)
# print((torch.randn(X.shape)<0.5).float())
# print(dropout(X,0.5))

#初始化，两个隐层，每层权值用mean=0,std=0.01的正态分布初始化,两个隐藏层的输出都是256
num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 28*28, 10, 256,256
w1 = torch.tensor(np.random.normal(0,0.01,size=(num_inputs,num_hiddens1)),dtype=torch.float,requires_grad=True)
w2 = torch.tensor(np.random.normal(0,0.01,size=(num_hiddens1,num_hiddens2)),dtype=torch.float,requires_grad=True)
w3 = torch.tensor(np.random.normal(0,0.01,size=(num_hiddens2,num_outputs)),dtype=torch.float,requires_grad=True)
b1 = torch.zeros(num_hiddens1,requires_grad=True)
b2 = torch.zeros(num_hiddens2,requires_grad=True)
b3 = torch.zeros(num_outputs,requires_grad=True)
params = [w1,b1,w2,b2,w3,b3]

drop_prob1,drop_prob2 = 0.2,0.5 #设置两个隐藏层的丢弃率，一般靠近输入层的要小一点
def net(X,is_training=True):
    X = X.view(-1,num_inputs)
    H1 = (torch.matmul(X,w1)+b1).relu() #每个隐层使用RELU函数非线性映射
    if is_training: #只需要在训练模型的时候才使用丢弃法（添加丢弃层）
        H1 = dropout(H1,drop_prob1)
    H2 = (torch.matmul(H1, w2) + b2).relu()  # 每个隐层使用RELU函数非线性映射
    if is_training:  # 只需要在训练模型的时候才使用丢弃法（添加丢弃层）
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2,w3)+b3

num_epochs,lr,batch_size = 5,100.0,256
loss = torch.nn.CrossEntropyLoss() #用于分类任务的交叉熵损失函数
train_iter, test_iter =d21.load_data_fashion_mnist(batch_size) #批量读取数据集到训练集和测试集
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
