import torch
# x = torch.ones(2,2,requires_grad=True)
# print(x)
# print(x.grad_fn)
# y = x+2
# print(y)
# print(y.grad_fn)
# print(x.is_leaf,y.is_leaf)
# z = y*y*3
# out = z.mean()
# print(z,out)
# x.requires_grad = False
# print(x.requires_grad)
# x.requires_grad_(True)
# print(x.requires_grad)

# out.backward()
# print(x.grad)
# out2 = x.sum()
# out2.backward()
# print(x.grad)
#
# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)

#求非标量的梯度
# a = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
# b = 2 * a
# c = b.view(2,2)
# print(c)
# v = torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
# c.backward(v)
# print(a.grad)

#中断梯度追踪
# x = torch.tensor(1.0,requires_grad=True)
# y1 = x**2
# with torch.no_grad():
#     y2 = x**3
# y3 = y1 + y2
# # print(x.requires_grad)
# # print(y1,y1.requires_grad)
# # print(y2,y2.requires_grad)
# # print(y3,y3.requires_grad)
# y3.backward()
# print(x.grad)

#修改tensor的值，又不希望被autograd记录，影响到反向传播，可以对tensor.data进行操作
x = torch.ones(1,requires_grad=True)
print(x.data)
print(x.data.requires_grad)

y = 2*x
x.data *= 100
y.backward()
print(x)
print(x.grad)