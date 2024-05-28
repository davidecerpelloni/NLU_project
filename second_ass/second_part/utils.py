import torch
import torch.utils.data as data
from collections import Counter
import json
from pprint import pprint
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 0
device = 'cpu' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def collate_fn2(data):
    #
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def collate_fn(batch):
    """
    Custom collate function to pad the sequences for BERT model inputs.
    """

    # Helper function to pad sequences in a batch
    def pad_sequences(sequences, pad_token=0):
        """
        Pad the sequences in the batch to the maximum sequence length.
        """
        return pad_sequence(sequences, batch_first=True, padding_value=pad_token)

    # Extracting each field from the batch
    #utterances = [torch.tensor(item['utterance']) for item in batch]
    slots = [item['slots'] for item in batch]
    intents = torch.tensor([item['intent'] for item in batch])
    input_ids = [item['utterance'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]

    # Padding sequences
    #padded_utterances = pad_sequences(utterances)
    padded_slots = pad_sequences(slots, pad_token=0)  # Assuming 0 is the pad token for slots
    padded_input_ids = pad_sequences(input_ids, pad_token=0)  # Assuming 0 is the pad token for input_ids
    padded_attention_masks = pad_sequences(attention_masks, pad_token=0)
    padded_token_type_ids = pad_sequences(token_type_ids, pad_token=0)

    # Creating the final dictionary to return
    batch_dict = {
        #'utterances': padded_utterances,
        'slots': padded_slots,
        'intents': intents,
        'utterance': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'token_type_ids': padded_token_type_ids
    }

    return batch_dict

class Lang():
    def __init__(self, tokenizer, words, intents, slots, cutoff=0):
        self.tokenizer = tokenizer
        self.word2id = {token: idx for idx, token in enumerate(self.tokenizer.vocab.keys())}
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        # Tokenize each word using the BERT tokenizer
        tokenized_words = [subword for word in elements for subword in self.tokenizer.tokenize(word)]
        
        # Count the frequency of each tokenized subword
        count = Counter(tokenized_words)
        
        # Create the vocab dictionary
        vocab = {'[PAD]': PAD_TOKEN}  # BERT's pad token is '[PAD]'
        if unk:
            vocab['[UNK]'] = len(vocab)  # BERT's unk token is '[UNK]'
        
        for token, freq in count.items():
            if freq > cutoff:
                vocab[token] = len(vocab)
        
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

class IntentsAndSlots2 (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        sample = {'utterance': utt}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
class TokenizeUtt(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, lang):
        self.tokenizer = tokenizer
        self.utterances = [sample['utterance'] for sample in dataset]
        self.slots = [sample['slots'] for sample in dataset]
        self.intents = [sample['intent'] for sample in dataset]
        
        # Create a slot label to ID mapping
        all_slots = set()
        for slot in self.slots:
            all_slots.update(slot.split())
        # self.slot2id = {}
        # self.slot2id['pad'] = len(self.slot2id)  # Add 'pad' to the mapping    
        # self.slot2id = {slot: idx for idx, slot in enumerate(sorted(all_slots))}
        self.slot2id = lang.slot2id
        self.id2slot = {idx: slot for slot, idx in self.slot2id.items()}
        
        # Create an intent to ID mapping
        # all_intents = set(self.intents)
        # self.intent2id = {intent: idx for idx, intent in enumerate(sorted(all_intents))}
        self.intent2id = lang.intent2id
        self.id2intent = {idx: intent for intent, idx in self.intent2id.items()}

    def __len__(self):
        return len(self.utterances)
    
    def get_slot2id(self):
        return self.slot2id

    def get_intent2id(self):
        return self.intent2id

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        slot = self.slots[idx]
        intent = self.intents[idx]
                
        words = utterance.split()
        slot_labels = slot.split()

        tokens = []
        label_ids = []

        for word, slot in zip(words, slot_labels):
            # Tokenize the word
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)

            # Assign the slot to the first token and 'pad' to the rest
            label_ids.append(self.slot2id[slot])
            label_ids.extend([self.slot2id['pad']] * (len(word_tokens) - 1))

        # Add [CLS] and [SEP] tokens and their corresponding slot labels
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [self.slot2id['pad']] + label_ids + [self.slot2id['pad']]

        # Convert tokens to token IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Token type IDs (not used for single-sequence inputs)
        token_type_ids = [0] * len(input_ids)
        
        intent_id = self.intent2id[intent]

        return {
            #'utterance': utterance,
            'slots': torch.tensor(label_ids, dtype=torch.long),
            'intent': torch.tensor(intent_id,dtype=torch.long),
            'utterance': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

  