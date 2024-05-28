import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size) #the softmax
        #adding dropout layer
        self.dropout = nn.Dropout(p=out_dropout, inplace=False)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        first = self.dropout(emb) #substitute with variational dropout
        lstm_out, _  = self.lstm(first)
        second = self.dropout(lstm_out) #substitute with variational dropout
        output = self.output(second).permute(0,2,1)
        return output