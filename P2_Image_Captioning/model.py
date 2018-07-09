# References: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
import random


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, deep=False):

        super(EncoderCNN, self).__init__()
        if deep:
            resnet = models.resnet152(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        rem = 1 # number of end layers to remove
        modules = list(resnet.children())[:-rem]
        self.resnet = nn.Sequential(*modules)

        # My modifications
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)

        # My modifications
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = F.relu(features)
        features = self.bn(features)

        return features


class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed = nn.Embedding(vocab_size, embed_size) ###
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True) ###
        self.lstm_state = None
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        if num_layers >= 2:
            self.lstm_two = nn.LSTM(hidden_size, hidden_size, num_layers=(num_layers-1), batch_first=True) 
            self.linear_two = nn.Linear(hidden_size, vocab_size)

        #self.init_weight()

    def forward(self, features, captions, use_teacher_forcing=False):
        self.lstm_state = None
        if use_teacher_forcing:
            return self.forward_1(features, captions)
        else:
            return self.forward_2(features, seq_len=captions.size()[1])

    def _forward(self, inputs):
        output, self.lstm_state = self.lstm(inputs, self.lstm_state)
        if self.num_layers >= 2: ### finish
            output, _ = self.lstm_two(output)
            #output = self.linear_two(output)
            output = self.linear(output)
        else:
            output = self.linear(output) # (batch_size,seq_len,embed_size)
        return output
        
    
    def forward_1(self, features, captions):
        output = self.embed(captions[:,:-1]) # last <end> not needed
        output = torch.cat((features.unsqueeze(1), output), 1)
        output = self._forward(output)
        return output.view(-1, self.vocab_size) 
    
    def forward_2(self, features, seq_len=20):
        #batch_size = captions.size()[0]
        inputs = features.unsqueeze(1)
        outputs = []
        for _ in range(seq_len):
            output = self._forward(inputs)
            #topi = output.topk(1)[1]
            #inputs = topi.squeeze().detach()
            #inputs = F.softmax(output, dim=2)
            inputs = F.softmax(output, dim=2).squeeze(1)
            inputs = torch.multinomial(inputs,1).squeeze()
            #inputs = torch.tensor(
            #    [np.random.choice(range(output.size(2)),p=inputs[b,0,:].detach().cpu().numpy()) for b in range(output.size(0))],
            #    dtype=torch.long,
            #    device=self.device)
            inputs = self.embed(inputs).unsqueeze(1)
            outputs.append(output)
        return torch.cat(outputs, dim=1).view(-1, self.vocab_size) # (batch_size,seq_len,embed_size)
    
    def sample(self, inputs, hiddens=[None,None], max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        hidden, hidden_two = hiddens
        tokens = []
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            if self.num_layers >= 2:
                output, hidden_two = self.lstm_two(output, hidden_two)
                #output = self.linear_two(output)
                output = self.linear(output)
            else:
                output = self.linear(output)
            _, target_index = output.max(2)
            tokens.append(target_index.detach().cpu().item())
            inputs = self.embed(target_index)
            #inputs = F.relu(inputs)
        return tokens

'''
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def normalize(self, v):
        v = np.array(v)
        v -= np.min(v)
        norm = sum(v)
        if norm == 0: 
            return v
        return v / norm
    
    def altsample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        return None
'''

'''    
    def init_hidden(self):
        """
        Initialize the hidden states of LSTM
        """
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


    def init_weight(self):
        """Initialize weights"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0.0)
'''