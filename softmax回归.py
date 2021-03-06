import torch
import torchvision #用于构建计算机视觉模型
#datasets 用于下载和加载数据的函数以及常用数据集接口
#models 包含常用的模型结构，包含预训练模型如AlexNet、VGG、ResNet等
import torchvision.transforms as transforms #包含常用的图片变换
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d21
import numpy as np

#transform.ToTensor() 使所有数据转换为Tensor(torch.float32且位于（0.0，1.0）的tensor)
# #获取训练集
# mnist_train = torchvision.datasets.FashionMNIST(root='',train=True,download=True,transform=transforms.ToTensor())
# #获取测试集
# mnist_test = torchvision.datasets.FashionMNIST(root='',train=False,download=True,transform=transforms.ToTensor())

# print(len(mnist_train),len(mnist_test))
# feature,label = mnist_train[0] #获取训练集第一个图片的内容
# # print(feature.shape,label)

# X,y = [],[]
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append((mnist_train[i][1]))
# d21.show_fashion_mnist(X,d21.get_fashion_mnist_labels(y))

import torch.utils.data as Data

batch_size = 256
# if sys.platform.startswith('win'): #若系统不是windows
#     num_workers = 0  #不用额外的进程来加速读取数据
# else:
#     num_workers = 4
train_iter,test_iter = d21.load_data_fashion_mnist(batch_size)

# start = time.time()
# for X,y in train_iter:
#     continue
# print('%.2f sec'%(time.time()-start))

num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad = True
b.requires_grad = True
# X = torch.rand((2,5))
# X_prob = d21.softmax(X)
# print(X_prob,X_prob.sum(dim=1))
def net(X):
    return d21.softmax(torch.mm(X.view((-1,num_inputs)),W)+b) #X.view()中的-1是指根据列自动计算行数

# y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# y = torch.LongTensor([0,2])
# y_hat.gather(1,y.view(-1,1))
# print(y)
# print(d21.accuracy(y_hat,y))

num_epochs,lr = 5,0.1
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params = None, lr = None, optimizer = None):
    for epoch in range(num_epochs):
        train_l_sum , train_acc_sum , n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d21.sgd(params,lr,batch_size)
            else:
                optimizer.step()#softmax回归的简洁实现一节将用到

            train_l_sum+=l.item() #每轮都累计损失
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item() #获取预测正确的次数
            n+=y.shape[0]
        test_acc=d21.evaluate_accuracy(test_iter,net)
        print('epoch {}, loss {}, train acc {}, test acc {}'.format(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))

train_ch3(net,train_iter,test_iter,d21.cross_entropy,num_epochs,batch_size,params = [W,b],lr=lr)

X,y=iter(test_iter).next()

true_labels = d21.get_fashion_mnist_labels(y.numpy())
pred_labels = d21.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [(true + '\n'+ pred) for true,pred in zip(true_labels,pred_labels)]
d21.show_fashion_mnist(X[0:9],titles[0:9])