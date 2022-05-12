import torch
import numpy as np
import sys
import d2lzh_pytorch as d21
#y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + e
n_train,n_test,true_w,true_b = 100,100,[1.2,-3.4,5.6],5
features = torch.randn((n_train+n_test,1))  #生成满足标准正态分布的样本集
#把x,x^2,x^3拼接到一起
poly_feature = torch.cat((features,torch.pow(features,2),torch.pow(features,3)),1)
#y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + e
labels = (true_w[0]*poly_feature[:,0] + true_w[1]*poly_feature[:,1] + true_w[2]*poly_feature[:,2] + true_b)
#噪声项
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

num_epochs,loss=100,torch.nn.MSELoss() #定义训练轮数和损失函数
def fit_plot(train_features,test_features,train_labels,test_labels):
    net = torch.nn.Linear(train_features.shape[-1],1)#linear中已经将参数初始化了，无需再初始化

    batch_size = min(10,train_labels.shape[0]) #批量大小取小于10的训练集个数
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)#将特征和标签合并成一个数据集
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)#批量读取数据集中的数据

    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    train_ls,test_ls = [],[] #记录训练集损失值和测试集损失值的数组
    for _ in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y.view(-1,1))
            optimizer.zero_grad()#backward之前梯度清零
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1,1)
        test_labels = test_labels.view(-1,1)
        train_ls.append(loss(net(train_features),train_labels).item())#用loss().item()是怕内存爆炸
        test_ls.append(loss(net(test_features),test_labels).item())#每次训练完就算一次总的损失
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    d21.semilogy_d21(range(1,num_epochs+1),train_ls,'epochs','loss',range(1,num_epochs+1),test_ls,
                         ['train','test'])
    print('weight:',net.weight.data,'\nbias:',net.bias.data)
#正常
# fit_plot(poly_feature[:n_train,:],poly_feature[n_train,:],labels[:n_train],labels[n_train:])
#欠拟合
fit_plot(features[:n_train],features[n_train:],labels[:n_train],labels[n_train:])
#过拟合
# fit_plot(poly_feature[0:2,:],poly_feature[n_train,:],labels[0:2],labels[n_train:])
