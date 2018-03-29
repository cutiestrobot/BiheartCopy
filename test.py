import torch
import torch.nn as nn
from MySetting import *
from torch.autograd.variable import Variable
import datetime


class testclass(nn.Module):
    def __init__(self, idd):
        super(testclass, self).__init__()
        self.idd = idd
        self.linear = nn.Linear(3, 5)

    def forward(self, x):
        self.idd += 1
        return self.linear(x)


    def save(self, path="G:\\pyfiles\\", name="sb.pkl"):
        torch.save(self.state_dict(), path+name)

'''
a = Variable(torch.FloatTensor([2, 4, 5.4]).view(1, 3))
tc = testclass(4)

print(tc.forward(a))

tc.save()

net2 = testclass(133)
print(net2.forward(a))


net2.load_state_dict(torch.load("G:\\pyfiles\\sb.pkl"))

print(net2.forward(a))
'''

#a = Variable(torch.FloatTensor([[1,2,33,4,5,6],[-1,2,-77,1,7,4]]).view(2, 3, 2))
#print(a)
#print(a.mean(dim=0))

count = len(open(wv_Path,'rU').readlines())
print(count)


