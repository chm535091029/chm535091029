from matplotlib import pyplot as plt
import random
import torch
import torch.utils.data as Data #提供了有关数据处理的工具
import torchvision #用于构建计算机视觉模型
import sys
import torchvision.transforms as transforms
import numpy as np
from torch import nn

def use_display(figsize=(10,5)):
    plt.show()

def data_iter(batch_size,features,labels): #读取全部小批量数据，用于批量梯度下降
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #打乱列表顺序
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)]) #最后一次可能不足一个batch
        yield features.index_select(0,j),labels.index_select(0,j)
        #.index_select(以哪个轴为标准，选取的行或列的列表)
def linreg(X,w,b): #线性回归公式
    return torch.mm(X,w) + b

def squard_loss(y_hat,y): #线性回归损失函数
    return (y_hat - y.view(y_hat.shape))**2/2

def sgd(params,lr,batch_size): #小批量梯度下降
    for param in params:
        param.data -= lr * param.grad/batch_size

def get_fashion_mnist_labels(labels):#将模型预测值转化为对应标签名
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images,labels): #分区画图展示softmax分类
    use_display()
    _,figs = plt.subplots(1,len(images),figsize=(12,12))#对图进行分区分割，返回分区的位置
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy()) #在每个分区上画图，参数为图或者像素数组
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
#获取并读取Fashion-MNIST数据集，返回训练集和测试集的批量梯度下降迭代器
def load_data_fashion_mnist(batch_size):
    # 获取训练集
    mnist_train = torchvision.datasets.FashionMNIST(root='', train=True, download=True, transform=transforms.ToTensor())
    # 获取测试集
    mnist_test = torchvision.datasets.FashionMNIST(root='', train=False, download=True, transform=transforms.ToTensor())
    if sys.platform.startswith('win'):  # 若系统不是windows
        num_workers = 0  #不用额外的进程来加速读取数据
    else:
        num_workers = 4
    #批量读取数据集
    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def softmax(X):#计算softmax值
    X_exp = X.exp()
    parition = X_exp.sum(dim=1,keepdim = True)
    return X_exp/parition  #这里用了广播机制

def cross_entropy(y_hat,y): #交叉熵损失函数
    return -torch.log(y_hat.gather(1,y.view(-1,1)))#真实标记列向量中每行真实标记的索引

def accuracy(y_hat,y):#准确率计算
    return (y_hat.argmax(dim=1)==y).float().mean().item()

def evaluate_accuracy(data_iter,net):#评价模型在数据集data_iter上的准确率
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n

#训练模型
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
            if optimizer is None: #设置优化器，默认是SGD
                sgd(params,lr,batch_size)
            else:
                optimizer.step()#softmax回归的简洁实现一节将用到

            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch {}, loss {}, train acc {}, test acc{}'.format(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))

#对x的形状转换
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)

#画对数图
def semilogy_d21(x_vals,y_vals,x_labels,y_labels,x2_vals=None,y2_vals=None,legend=None,figsize=(3.5,2.5)):
    plt.figure(figsize=figsize)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.semilogy(x_vals,y_vals,label=legend[0])#半对数图
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':',label=legend[1])#半对数图
        plt.legend()
    plt.show()
