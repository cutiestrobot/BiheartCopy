import time
import math
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import MyFunc as mf
import Myclass as mc
from MySetting import *




# language preparation
lines = [
    ["this is the best way to run.", "run"],
    ["rabbit has two ears and one eye.", "rabbit's appearance."],
    ["big fish cannot fly.", "big"]
]
lang = mc.Lang("lang")
for line in lines:
    lang.index_words(line[0])

print("reading word vector ...")
wv = mc.wordVector(wv_Path, lang.word2index, size=10000, dim=300)
print("finish.")

vocab_size = lang.n_words
h_size = wv.dim



# reading net
print("reading enco, deco...")
enco = mc.Encoder(vocab_size, h_size, wv.veclist, n_layers=2)
deco = mc.AttnDecoder(hidden_size=h_size, output_size=vocab_size, n_layers=2)

enco.load_para(net_Path, "hh.pkl")
deco.load_para(net_Path, "sb.pkl")

print("finish.")


# training net

torch.manual_seed(2333)

X = Variable(torch.LongTensor([
    [0,3,2,10,2,9,1],
    [0,9,4,10,2,7,1],
    [0,5,9,10,6,3,1]]).view(3,1,-1))
Y = Variable(torch.LongTensor([
    [4,3,11,3,10],
    [10,5,11,3,8],
    [6,10,11,7,4],
]).view(3,1,-1))
if USE_CUDA:
    enco.cuda()
    deco.cuda()
    X = X.cuda()
    Y = Y.cuda()

enco_opt = optim.Adam(enco.parameters(), lr=0.001)
deco_opt = optim.Adam(deco.parameters(), lr=0.001)
criterion = nn.NLLLoss()

print("start training...")
for epoch in range(50):
    for pair in range(len(Y)):
        loss = mf.train(
            X[pair].squeeze(),
            Y[pair].squeeze(),
            enco, deco,
            enco_opt, deco_opt,
            criterion=criterion,
            clip=5.0
        )
        if epoch % 5 == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "loss=", loss
                  )
# save net
print("saving networks...")
enco.save_para(net_Path, "hh.pkl")
deco.save_para(net_Path, "sb.pkl")
print("done.")
