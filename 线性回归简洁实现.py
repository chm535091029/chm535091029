import torch.utils.data as Data #提供了有关数据处理的工具
import torch
import numpy as np
from torch.nn import init #定义了各种初始化方法
import torch.nn as nn #定义了大量神经网络的层
import d2lzh_pytorch
#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1,(num_examples, num_inputs)))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))

batch_size = 10
dataset = Data.TensorDataset(features,labels)#将训练数据的特征和标签组合成数据集
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)#随机读取小批量，返回X和y

# for X,y in data_iter:
#     print(X,y)
#     break
# class LinearNet(nn.Module):
#     def __init__(self,n_feature):#特征数
#         super(LinearNet,self).__init__()
#         self.linear = nn.Linear(n_feature,1) #nn.Linear(输入特征数,输出数）
#     #forward 定义前向传播
#     def forward(self,x):
#         y = self.linear(x)
#         return y
# net = LinearNet(num_inputs)

#写法一：
# net = nn.Sequential(nn.Linear(num_inputs,1),nn.Linear(num_inputs,2))
# 写法二：
net = nn.Sequential() #Sequential是一个有序的容器
net.add_module('linear',nn.Linear(num_inputs,1))
# print(net[2])
#写法三：
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([('linear',nn.Linear(num_inputs,1))]))
# print(net)

init.normal_(net[0].weight,mean=0,std=0.01) #权重参数每个元素初始化为随机采样于均值为0方差为0.01的正态分布
init.constant_(net[0].bias,val=0)#修改偏差为0

loss = nn.MSELoss() #定义均方误差损失函数

import torch.optim as optim #优化算法的包 如SGD、Adam和RMSProp等。
optimizer = optim.SGD(net.parameters(),lr = 0.03) #实现小批量下降的优化算法、学习率为0.03
#左边参数可以是一个多个字典的列表，包含了子层和其学习率，若字典内没有"lr"键，就默认使用最外面的lr=
# optimizer = optim.SGD([{'params':net.subnet1.parameters(),"lr":0.01}],lr = 0.03)
# print(optimizer)

#调整学习率
for param_group in optimizer.param_groups:
    param_group["lr"]*=0.1

num_epochs = 50
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = net(X.to(torch.float32))
        l = loss(output,y.view(-1,1).to(torch.float32))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward() #计算梯度
        optimizer.step()

    print('eopch %d,loss:%f'%(epoch,l.item()))
print(true_w,net[0].weight)  #网络结点的权重
print(true_b,net[0].bias)#方差