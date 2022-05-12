import torch
import numpy as np
import sys
import d2lzh_pytorch as d21

batch_size = 256
train_iter,test_iter = d21.load_data_fashion_mnist(batch_size)#加载训练集和测试集

num_inputs,num_outputs,num_hiddens = 784,10,256
W1 = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)
W2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
b2 = torch.zeros(num_outputs,dtype=torch.float)

params = [W1,b1,W2,b2]
for param in params:
    param.requires_grad = True

def relu(X): #手动定义Relu函数
    return torch.max(input=X,other=torch.tensor(0.0))

def net(X):
    X = X.view(-1,num_inputs)
    H = relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H,W2) + b2

num_epochs,lr = 5,100.0
loss = torch.nn.CrossEntropyLoss()
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
