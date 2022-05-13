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

def init_params():#初始化参数
    w = torch.randn((num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w): #L2范数惩罚项
    return (w**2).sum()/2

batch_size,num_epochs,lr = 1,100,0.003
net,loss = d21.linreg,d21.squard_loss  #线性回归和均方误差

dataset = torch.utils.data.TensorDataset(train_features,train_labels)#把标签和特征和并为每一个样本
train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

def fit_plot(lambd):
    w,b = init_params()
    train_ls,test_ls = [],[] #记录每个周期的损失
    for _ in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X,w,b),y) + lambd*l2_penalty(w) #添加了L2范数惩罚项
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_() #backward前梯度清零
                b.grad.data.zero_()
            l.backward()
            d21.sgd([w,b],lr,batch_size)  #L2范数改变了loss但优化器本身未变
        train_ls.append(loss(net(train_features,w,b),train_labels).mean().item()) #每轮记录损失到数组
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d21.semilogy_d21(range(1,num_epochs+1),train_ls,'epochs','loss',range(1,num_epochs+1),test_ls,['train','test'])
    print('L2 norm of w:',w.norm().item())
fit_plot(lambd=0) #λ等于0，表示不加l2范数，直接查看过拟合影响

