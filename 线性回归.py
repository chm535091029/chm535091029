# 导包
# %matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
import random
import torch
import numpy as np
import d2lzh_pytorch

#向量相加的方法
# a = torch.ones(1000)
# b = torch.ones(1000)
# start = time() #获得当前时间
# c =torch.zeros(1000)
# for i in range(1000):  #方法1
#     c[i] = a[i] + b[i]
#
# d = a + b #方法二  更省时
# print(time()-start)
#
# a = torch.ones(3)
# b = 10
# print(a+
#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1,(num_examples, num_inputs)))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))


# def use_svg_display():
#     #用矢量图显示
#     matplotlib_inline.set_matplotlib_formats('svg')

# def set_figsize(figsize = (3.5,2.5)):
#     #设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize


# plt.scatter(features[:,1].numpy(),labels.numpy(),1)
# use_display((10,5))

batch_size = 10

# for X, y in d2lzh_pytorch.data_iter(batch_size,features,labels):
#     print(X,y)
#     break
w= torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32) #将权重初始化成均值为0，标准差为0.01的正太随机数
b = torch.zeros(1,dtype=torch.float32)
w.requires_grad = True
b.requires_grad = True

lr = 0.03 #学习率
num_epochs = 3 #迭代周期个数
# net = d2lzh_pytorch.linreg
# loss = d2lzh_pytorch.squard_loss(net,labels)
for epoch in range(num_epochs):
    #在每一个迭代周期中，会使用训练数据集中所有样本一次
    #x和y分别是小批量样本的特征和标签
    for X,y in d2lzh_pytorch.data_iter(batch_size,features,labels):
        l = d2lzh_pytorch.squard_loss(d2lzh_pytorch.linreg(X.to(torch.float32),w,b),y).sum() #有关小批量x和y的损失
        l.backward() #小批量的损失对模型求参数梯度
        d2lzh_pytorch.sgd([w,b],lr,batch_size)#使用小批量随机梯度下降迭代模型参数
        #不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_1 = d2lzh_pytorch.squard_loss(d2lzh_pytorch.linreg(features.to(torch.float32),w,b),labels)
    print('epoch%d,loss %f' % (epoch + 1,train_1.mean().item()))
print(true_w,"\n",w)
print(true_b,"\n",b)

