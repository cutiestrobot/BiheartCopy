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
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

obj_file = shelve.open(shelve_Path+'myshelve')
lang = obj_file['lang']
obj_file.close()

enco = mc.Encoder(lang.n_words, h_size, pretrained=None, n_layers=1)
deco = mc.AttnDecoder(hidden_size=h_size, output_size=lang.n_words, n_layers=1)
enco.load_para(net_Path, "mini_Enco.pkl")
deco.load_para(net_Path, "mini_Deco.pkl")
enco_opt = optim.Adam(enco.parameters(), lr=0.0001)
deco_opt = optim.Adam(deco.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

start_time = time.time()

#loading data you need
#create the loading_plan
loading_plan=[]
temp=1;
for i in range(1018):
    loading_plan.append((temp,temp+5000))
    temp=temp+5001
loading[1017]=(5086018,5089618);


for epoch in range(2):
    piece_cnt=0
    random.shuffle(loading_plan)
    for start, end in loading_plan:
        piece_cnt=piece_cnt+1
        giga = mc.GigaLoader(start, end)
        print("now loading ",piece_cnt,"piece/1018 total piseces")
        train = []
        for pair in giga.lines:
            xi = [lang.word2index[x] for x in pair[0]]
            yi = [lang.word2index[y] for y in pair[1]]
            train.append([xi, yi])

        print("start training... from ", start, " to ", end)

        for pair in train:
            X = Variable(torch.LongTensor(pair[0]))
            Y = Variable(torch.LongTensor(pair[1]), volatile=True)
            if USE_CUDA:
                enco.cuda()
                deco.cuda()
                X = X.cuda()
                Y = Y.cuda()
            loss, predict = mf.train(
                X.squeeze(),
                Y.squeeze(),
                enco, deco,
                enco_opt, deco_opt,
                criterion=criterion,
                clip=5.0
             )
            # print('')
            # mf.print_row(predict, lang)



            if time.time() - start_time > SAVE_TIME:
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      "loss=", loss)
                print("saving networks...")
                enco.save_para(net_Path, "mini_Enco.pkl")
                deco.save_para(net_Path, "mini_Deco.pkl")
                print("done.")
                start_time = time.time()

    print('finish all epochs of this piece')

    #log.close()







