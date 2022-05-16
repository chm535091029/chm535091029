import torch
from torch import nn

class MLP(nn.Module):
    #声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self,**kwargs):
        #调用父类的构造函数进行必要的初始化，这样构造示例的时候还可以指定其他函数
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256,10)
    #定义模型的前向计算，即如何根据输入X计算输出
    def forward(self,x):
        a = self.act(self.hidden(x))
        return self.output(a)
x = torch.rand(2,784)
# net = MLP()
# print(net(x))

#Sequential类的手动实现
from collections import OrderedDict
class MySequential(nn.Module):
    def __init__(self,*args):
        super(MySequential,self).__init__()
        if len(args)==1 and isinstance(args[0],OrderedDict): #如果传入的是一个OrderedDict实例
            for key, module in args[0].items():
                self.add_module(key,module) #add_module方法会将module添加进self._module(一个OrderedDict)
        else: #传入的是一些module
            for idx,module in enumerate(args):
                self.add_module(str(idx),module)
    def forward(self,input):
        for module in self._modules.values(): #_module返回一个OrderedDict，依次保证顺序进行计算
            input = module(input)
        return input
# net = MySequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10),)
# print(net(x))

#ModuleList类
net = nn.ModuleList([nn.Linear(784,256),nn.ReLU()]) #传入列表
net.append(nn.Linear(256,10)) #类似List的append操作
# del net[-1]
# print(net)

#ModuleDict类
net = nn.ModuleDict({'linear':nn.Linear(784,256),'act':nn.ReLU()})
net['output'] = nn.Linear(256,10) #类似字典地添加键值对
# print(net['linear'])
# print(net.output) #类似字典地访问键值对

#利用继承Module类创建比较复杂的模型
class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        #创建不可训练的常数参数
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
        # self.linear.weight.requires_grad=False
    def forward(self,x):
        x = self.linear(x)
        #使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x,self.rand_weight.data)+1)
        #复用全连接层，等价于两个全连接层共享参数
        x = self.linear(x)
        #控制流，这里需要调用item函数来返回标量进行比较
        while x.norm().item() >1:
            x/=2
        if x.norm().item()<0.8:
            x*=10

        return x.sum()
X = torch.rand(2,20)
net = FancyMLP()
print(net)
print(net(X))

class NestMLP(nn.Module):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30),nn.ReLU())
    def forward(self,x):
        return self.net(x)

net = nn.Sequential(NestMLP(),nn.Linear(30,20),FancyMLP())

# X = torch.rand(2,40)
# print(net(X))