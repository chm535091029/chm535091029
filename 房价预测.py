import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import d2lzh_pytorch

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor) #设置张量的默认形式

#读取训练集和测试集
train_data = pd.read_csv('house_price/train.csv')
test_data = pd.read_csv('house_price/test.csv')
# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
#将训练数据和测试数据合在一起
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#标准化
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index #非对象类型的索引
#对每个元素使用apply中的函数进行转换
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
#标准化后每个特征均值变成0,可以用0代替缺失值
all_features = all_features.fillna(0)
#将离散值也转化为指示特征，即把离散值转化成各种的标志位,dummy_na表示是否将缺失值也当作合法的特征值并为其创建指示特征（标志位）
all_features = pd.get_dummies(all_features,dummy_na=True)
#通过values属性得到NUMPY格式的数据，并转换成ADArray方便后面的训练。
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values,dtype=torch.float).view(-1,1)
#使用基本的线性回归和平方损失函数来训练
loss = torch.nn.MSELoss() #平方损失函数
def get_net(feature_num):#定义线性回归网络
    net = nn.Linear(feature_num,1)
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)
    return net
#定义对数均方根误差
def log_rmse(net,features,labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features),torch.tensor(1.0)) #将小于1的值设为1，使得取对数时的数值更稳定
        rmse = torch.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.item()
#定义训练函数，使用了Adam优化算法
def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    dataset = torch.utils.data.TensorDataset(train_features,train_labels) #把特征和标签合并成样本
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True) #批量读取数据集
    #使用Adam算法,用权重衰减，weight_decay是L2范式中的λ参数
    optimizer = torch.optim.Adam(params=net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X.float()),y.float())  #每轮训练采用均方误差
            optimizer.zero_grad() #反向传播前梯度清零
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels)) #每轮结束统计对数均方误差，不参与优化（不算梯度）
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls
#K折交叉验证，用于选择模型设计并调节超参数，下面函数返回第i折交叉验证所需要的训练和验证数据
def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0]//k
    X_train,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part = X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid = X_part,y_part #把第i折数据提出来当验证集
        elif X_train is None:
            X_train,y_train = X_part,y_part
        else:
            X_train = torch.cat((X_train,X_part),dim=0)  #把剩下的数据缝合在一起当验证集
            y_train = torch.cat((y_train,y_part),dim=0)
    return X_train,y_train,X_valid,y_valid
#K折交叉验证
def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum = 0,0
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train) #获取第i折的训练数据和验证数据
        net = get_net(X_train.shape[1])
        #在每一折上训练
        train_ls , valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        #累计每一轮的训练和验证误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # d2lzh_pytorch.semilogy_d21(range(1,num_epochs+1),train_ls,'epochs','rmse',range(1,num_epochs+1),
        #                                valid_ls,['train','valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i,train_ls[-1],valid_ls[-1]))
    return train_l_sum/k,valid_l_sum/k

k,num_epochs,lr,weight_decay,batch_size = 5,100,5,0,64
train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print('%d-fold validation:avg train rmse %f, avg valid rmse %f' %(k,train_l,valid_l))
#定义预测函数，在预测之前，使用完整的训练数据集来重新训练模型，并将预测结果存成提交所需要的格式
def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,
                   lr,weight_decay,batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2lzh_pytorch.semilogy_d21(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',legend=['train'])
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy() #把测试数据的特征带进模型预测
    test_data['SalePrice'] = pd.Series(preds.reshape(-1,1)[0]) #给原来的测试集再加一列房价预测结果
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=0)
    submission.to_csv('1.csv',index=False)

# train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
