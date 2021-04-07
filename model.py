import torch
import torch.nn as nn
from src.config import *

class RNNSentiment(nn.Module):
    def __init__(self,num_layer, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(RNNSentiment, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.LSTM = nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim,num_layers=self.num_layer,batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_dim,1)
        self.sig = nn.Sigmoid()

    def forward(self,x,hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out , hidden = self.LSTM(embeds,hidden)

        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)

        sig_out = sig_out.view(batch_size,-1)
        sig_out = sig_out[:,-1]

        return sig_out,hidden

    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layer, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.num_layer, batch_size, self.hidden_dim).zero_().cuda())
        return hidden


