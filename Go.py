import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import time
import MyFunc as mf
import Myclass as mc
from MySetting import *
import shelve
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

obj_file = shelve.open(shelve_Path+'myshelve')
lang = obj_file['lang']
obj_file.close()

enco = mc.Encoder(lang.n_words, h_size, pretrained=None, n_layers=2, )
deco = mc.AttnDecoder(hidden_size=h_size, output_size=lang.n_words, n_layers=2)
enco.load_para(net_Path, "mini_Enco.pkl")
deco.load_para(net_Path, "mini_Deco.pkl")
enco_opt = optim.Adam(enco.parameters(), lr=0.001)
deco_opt = optim.Adam(deco.parameters(), lr=0.001)
criterion = nn.NLLLoss()

start_time = time.time()

#loading data you need

loading_plan = [(100, 15000), (15001, 21000), (21001, 25000), (25001, 30000), (30001, 35000), (36000, 41000)]:
# loading_plan = [(3,7),(9,13),(15,25)]

for start, end in loading_plan:
    giga = mc.GigaLoader(start, end)

    train = []
    for pair in giga.lines:
        xi = [lang.word2index[x] for x in pair[0]]
        yi = [lang.word2index[y] for y in pair[1]]
        train.append([xi, yi])

    log = open('log.txt', 'a')
    print("start training... from ", start, " to ", end, file=log)

    for epoch in range(2):
        loss = 0
        for pair in train:
            X = Variable(torch.LongTensor(pair[0]))
            Y = Variable(torch.LongTensor(pair[1]))
            if USE_CUDA:
                enco.cuda()
                deco.cuda()
                X = X.cuda()
                Y = Y.cuda()
            loss = mf.train(
                X.squeeze(),
                Y.squeeze(),
                enco, deco,
                enco_opt, deco_opt,
                criterion=criterion,
                clip=5.0
             )
        if epoch % 1 == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "loss=", loss, file=log
                  )
        if time.time() - start_time > SAVE_TIME:
            print("saving networks...", file=log)
            enco.save_para(net_Path, "mini_Enco.pkl")
            deco.save_para(net_Path, "mini_Deco.pkl")
            print("done.", file=log)
            start_time = time.time()

    print('finish all epochs of this piece', file=log)
    log.close()







