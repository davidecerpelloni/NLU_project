import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

PAD_TOKEN = 0

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True) #bidirectional=True
        self.slot_out = nn.Linear(2*hid_size, out_slot)
        self.intent_out = nn.Linear(2*hid_size, out_int)
        self.utt_to_slot = nn.Linear(hid_size*2,hid_size)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.5)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = self.dropout(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        #packed_output = self.dropout(packed_output)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        #last_hidden = last_hidden[-1,:,:]

        #last_hidden_layer_minus_1 = last_hidden[-1, :, :]
        #last_hidden_layer_minus_2 = last_hidden[-2, :, :]

        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=1)

        utt_encoded = self.dropout(utt_encoded)

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        #to_slots = self.utt_to_slot(utt_encoded)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss, slots(a,b,c) => .permute(a,c,b) => slots(a,c,b)
        # Slot size: batch_size, classes, seq_len
        return slots, intent