import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import d2lzh_pytorch as d21

#绘图函数
def xyplot (x_vals,y_vals,name):
    plt.figure(figsize=(5,2.5))

    plt.xlabel('x')
    plt.ylabel(name+'(x)')

    plt.plot(x_vals.detach().numpy(),y_vals.detach().numpy())
    plt.show()

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
# y = x.relu() #使用relu函数作为激活函数
# xyplot(x,y,'relu')

# y = x.sigmoid() #使用sigmoid函数作为激活函数
# xyplot(x,y,'sigmoid')

y = x.tanh()
xyplot(x,y,'tanh')
# x.grad.zero_()
y.sum().backward() #对x的变量求导 ，因为y不是标量，因此需要对其用sum（）
xyplot(x,x.grad,'grad')