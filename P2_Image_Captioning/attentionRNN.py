import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
import random

class AttnEncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, deep=False, n_layers=1, dropout_p=0.1):
        super(AttnEncoderRNN, self).__init__()
        if deep:
            resnet = models.resnet152(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        rem = 1 # number of end layers to remove
        modules = list(resnet.children())[:-rem]
        self.resnet = nn.Sequential(*modules)
        self.input_size = resnet.fc.in_features
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.sentence = torch.tensor(list(range(self.input_size)), dtype=torch.long).view(1, -1)
        self.register_buffer('sentence_const', self.sentence)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.v = nn.Parameter(torch.rand(self.hidden_size).unsqueeze(0).unsqueeze(1), requires_grad=True)
        #stdv = 1. / math.sqrt(self.v.size(0))
        #self.v.data.normal_(mean=0, std=stdv)
        #self.embedding = nn.Embedding(self.input_size,self.embed_size, max_norm=1, norm_type=2)
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        #self.linear = nn.Linear(resnet.fc.in_features, self.hidden_size)
        #self.bn = nn.BatchNorm1d(self.hidden_size, momentum=0.01)
        #self.bidir = False
        #self.gru = nn.GRU(
        #    self.embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidir) ### embed_size

    def forward(self, images, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded_weights = self.resnet(images).squeeze(3)
        #features = self.resnet(images)
        #embedded_weights = features.squeeze(3)
        #print(embedded_weights.size())
        #print(self.v)
        #embedded_weights = torch.mul(embedded_weights, self.v.expand(embedded_weights.size(0),-1,-1)) # 
        #print(embedded_weights.size())
        embedded_weights = embedded_weights.expand(-1,-1,self.hidden_size)
        embedded = self.embedding(self.sentence_const)
        embedded = embedded.expand(embedded_weights.size(0), -1, -1)
        embedded = torch.mul(embedded_weights, embedded).transpose(0,1)
        embedded = self.dropout(embedded)
        # My modifications
        #features = features.view(features.size(0), -1)
        #features = self.linear(features)
        #features = self.bn(features).view(1,embedded_weights.size(0),self.hidden_size)
        return embedded, None
        # embedded.mean(dim=0, keepdim=True)
        #return embedded, features
        #outputs, hidden = self.gru(embedded, hidden) ###
        #if self.bidir:
        #    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        #return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.expand(max_len,-1,-1).transpose(0,1) ####
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.expand(encoder_outputs.data.shape[0],-1).unsqueeze(1) #[B*1*H] # repeat ,1)
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define layers
        self.i = nn.Parameter(torch.rand(self.hidden_size).unsqueeze(0).unsqueeze(1), requires_grad=True)
        stdv = 1. / math.sqrt(self.i.size(0))
        self.i.data.normal_(mean=0, std=stdv)
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size * 1, output_size) #### Bidirectional removed 2 --> 1

    def forward(self, hidden, encoder_outputs, captions, use_teacher_forcing=True):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        outputs = []
        attentions = []
        word_input = captions[:,0]
        if not hidden:
            hidden = self.i.repeat(1,captions.size()[0],1)
        for i in range(captions.size()[1]-1):
            
            # Get the embedding of the current input word (last output word)
            word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
            word_embedded = self.dropout(word_embedded)
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
            context = context.transpose(0, 1)  # (1,B,V)
            # Combine embedded input word and attended context, run through RNN
            #print(word_embedded.size(),context.size())
            rnn_input = torch.cat((word_embedded, context), 2)
            #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
            output, hidden = self.gru(rnn_input, hidden)
            output = output.squeeze(0)  # (1,B,V)->(B,V)
            # context = context.squeeze(0)
            # update: "context" input before final layer can be problematic.
            # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
            #print(output.size())
            output = F.log_softmax(self.out(output))
            # Return final output, hidden state
            outputs.append(output)
            attentions.append(attn_weights)
            if use_teacher_forcing:
                word_input = captions[:,i+1]
            else:
                _, topi = output.topk(1)
                word_inputs = topi.squeeze().detach()
        return torch.cat(outputs, dim=1).view(-1, self.output_size), attentions
    
    def sample(self, hidden, encoder_outputs, max_len=20, SOS_token=0):
        dummy_caption = torch.tensor([[SOS_token]*max_len], device=self.device)
        outputs, attentions = self.forward(
            hidden, encoder_outputs, dummy_caption, use_teacher_forcing=False)
        return outputs, attentions
    
    
    
    