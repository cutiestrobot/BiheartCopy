import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import nltk
from MySetting import *

class GigaLoader:
    def __init__(self, start_line, end_line):
        self.start_line = start_line
        self.end_line = end_line
        self.lines = []

        if start_line < 0 or end_line > 500000:
            raise Exception("giga read too many lines!")

        # load to self.line as [  line[ line0[ 'wo','wd'], line1] ]
        with open(giga_Path) as file:
            for _ in range(start_line):
                file.readline()
            for index in range(start_line, end_line + 1):
                line = file.readline().strip('\n').lower()
                if line.count('\t') < 2:
                    continue
                line = line.split('\t')
                line.pop(0)
                for i, sent in enumerate(line):
                    line[i] = ("SOS " + sent + " EOS").split()
                # process the paragraph part
                for i, word in enumerate(line[0]):
                    try:
                        float(word)
                    except ValueError:
                        pass
                    else:
                        line[0][i] = '#'
                # process the title part
                for i, word in enumerate(line[1]):
                    try:
                        float(word)
                    except ValueError:
                        pass
                    else:
                        line[1][i] = '#'

                self.lines.append(line)
        # [
        #     [[],[]]
        # ]



class Lang:
    '''
    using words or sentences to build vocabulary

    '''

    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in nltk.word_tokenize(sentence):
            self.index_word(word)

    def index_word(self, word):
        word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_giga(self, giga):
        for item in giga.lines:
            for sect in item:
                for word in sect:
                    self.index_word(word)


class wordVector:
    # this is a outer vector loader
    def __init__(self, path, word2index, size=10000, dim=300):

        self.dim = dim
        self.veclist = np.zeros((len(word2index), dim))

        if size > 368990:
            raise Exception("In wordvec, when reading lexvec, size should be no larger than 368990")
        with open(path, encoding="utf-8") as f:
            for _ in range(size):
                line = f.readline().split(" ")
                word = line[0]
                if word in word2index:
                    vec = line[1: 1 + dim]
                    self.veclist[word2index[word]] = vec
                else:
                    continue


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained, n_layers=1):
        '''
        :param input_size: vocab size
        :param hidden_size: vector dim
        :param pretrained: numpy ndarray [vocabsize, vector dim]
        :param n_layers:
        '''

        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)

        if pretrained is not None:
            self.pre_word_embeds = True
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained))

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=False)

    def forward(self, word_inputs, hidden):
        '''
        :param word_inputs: a one dimensional variable
        :param hidden: hidden you init
        :return: output[seq,1,hidden_size] hidden[1,hidden_size]
        '''

        seq_len = len(word_inputs)
        embedded = (self.embedding(word_inputs).view(seq_len, 1, -1)).float()

        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def save_para(self, path, name):
        torch.save(self.state_dict(), path + name)

    def load_para(self, path, name):
        self.load_state_dict(torch.load(path+name))


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: [1,hidden_size] gru's output
        :param encoder_outputs: [seq, 1, hidden_size]
        :return: shape is [1, 1, seqlen]
        '''

        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))

        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies. seq_len => 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0).view(1, 1, -1)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)  # hidden_size => hidden_size Linear net
        energy = hidden.dot(energy)
        return energy


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        '''
        :param hidden_size: encoder's hidden_size
        :param output_size: vocab size
        :param n_layers: prepare proper init hidden yourself!
        :param dropout_p:
        '''
        super(AttnDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)  # s*row*col=1*1*vecdim(hidden)
        self.out = nn.Linear(hidden_size * 2, output_size)

        self.attn = Attn(hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        '''
        :param word_input: tensor of size 1. mostly a long int
        :param last_context: [1, hidden_size]
        :param last_hidden: [1, 1, hidden]
        :param encoder_outputs: [1, vocab_size]
        :return:
        '''
        # last_hidden is firstly the hidden of encoder
        # this func is called only once, as a single word step
        # an outer loop is needed
        # last_context: 1 * hidden

        word_embedded = self.embedding(word_input).view(1, 1, -1)  # => 1* 1* hidden
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)  # 1* 1* 2hidden

        rnn_output, hidden = self.gru(rnn_input, last_hidden)  # 1* 1* hidden, 2* 1* hidden

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)  # 1* 1* seq

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)  # (S=1) x B x N -> B x N
        context = context.squeeze(1)  # B x (S=1) x N -> B x N

        rawout = self.out(torch.cat((rnn_output, context), dim=-1))

        output = F.log_softmax(rawout, dim=-1)

        return output, context, hidden, attn_weights

    def copyinit_hidden(self, mode):
        '''
        :param mode: [n_layer(EN), 1, hidden_size]
        :return: [n_layer(DE), 1, hidden_szie]
        '''
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if len(mode) == self.n_layers:
            hidden = mode
        else:
            mode_mean = mode.mean(dim=0, keepdims=True)
            for i in range(self.n_layers):
                hidden[i] = mode_mean
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden

    def save_para(self, path, name):
        torch.save(self.state_dict(), path + name)

    def load_para(self, path, name):
        self.load_state_dict(torch.load(path+name))



