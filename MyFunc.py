import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from MySetting import *

def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, clip=5.0):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0                                                 # added onto for each word
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    ##############################

    # decoder's preparation
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))  # context is a 1*hiddensize zeros!
    decoder_hidden = decoder.copyinit_hidden(encoder_hidden)  # this is the last hidden 1*1*hidden_size

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        # model is already cudaed outside

    # start decoder
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention \
            = decoder(decoder_input, decoder_context,
                      decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])
        ## what is the shape of loss?

        topv, topi = decoder_output.data.topk(1)  # value and index of the top 1 max. topi is longtensor 1*1

        decoder_input = Variable(topi)

        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Stop at end of sentence (not necessary when using known targets)
        if topi[0][0] == EOS_token:
            break

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

