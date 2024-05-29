import os
import json
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from utils import *
from model import *
from functions import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim

current_dir = os.getcwd()

# Print the directory
print(f"Debugger working directory: {current_dir}")

#device = torch.device("cpu") #'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

# tmp_train_raw = load_data(os.path.join('second_ass','second_part','dataset','ATIS','train.json'))
# test_raw = load_data(os.path.join('second_ass','second_part','dataset','ATIS','test.json'))

print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))

pprint(tmp_train_raw[0])

# First we get the 10% of the training set, then we compute the percentage of these examples

portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
#print(len(intents)) 4978
count_y = Counter(intents) #group label:number_occurency

labels = []
inputs = []
mini_train = []

#intents = 0,flight 1,ground_service ...
for id_y, y in enumerate(intents):
    # If some intents occurs only once, we put them in training
    if count_y[y] > 1:
        #append utterance(espressione)
        inputs.append(tmp_train_raw[id_y])
        #array labels
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])
# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

# Intent distributions
print('Train:')
pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())}) #round(v/len(y_train),3)*100 v/len, tiene i primi 3 decimali, moltiplica per 100
print('Dev:'),
pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})  #print(Counter(y_train).items())   dict_items([('airfare', 381), ('flight', 3299),...])
print('Test:')
pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
print('='*89)
# Dataset size
print('TRAIN size:', len(train_raw))
print('DEV size:', len(dev_raw))
print('TEST size:', len(test_raw))

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not want unk labels,
                                        # however this depends on the research purpose

slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

# Create our datasets
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

lang = Lang(tokenizer, words, intents, slots, cutoff=0)

train_tokenized = TokenizeUtt(train_raw,tokenizer, lang)
dev_tokenized = TokenizeUtt(dev_raw,tokenizer, lang)
test_tokenized = TokenizeUtt(test_raw,tokenizer, lang)
    
print(train_tokenized[0])
print(dev_tokenized[0])
print(test_tokenized[0])

# dataset_tokenized = TokenizeUtt(corpus,tokenizer, lang)

# slot2id = dataset_tokenized.get_slot2id()
# intent2id = dataset_tokenized.get_intent2id()

# print(len(slot2id),slot2id) #130
# print(len(intent2id),intent2id) #26
# exit()

# Dataloader instantiations
train_loader = DataLoader(train_tokenized, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_tokenized, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_tokenized, batch_size=64, collate_fn=collate_fn)


hid_size = 250
emb_size = 300

lr = 9e-5 # learning rate (9e-5 overfitta)
clip = 5 # Clip the gradient

#model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
model = JointBERT(lang.intent2id,lang.slot2id).to(device)
#model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

n_epochs = 80
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
loop = tqdm(range(1,n_epochs))
for x in loop:
    loss = train_loop(train_loader, optimizer, criterion_slots,
                      criterion_intents, model, device, clip=clip)
    
    if x % 5 == 0: # We check the performance every 5 epochs
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                        criterion_intents, model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())

        f1 = results_dev['total']['f']
        loop.set_postfix(f1=f1,loss_dev=losses_dev[-1],intent=intent_res,loss_train=losses_train[-1])
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        if f1 > best_f1:
            best_f1 = f1
            # Here you should save the model
            patience = 3
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                         criterion_intents, model, lang)
print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])

plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Train and Dev Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(sampled_epochs, losses_train, label='Train loss')
plt.plot(sampled_epochs, losses_dev, label='Dev loss')
plt.legend()
plt.savefig('train_loss_12e.png')
plt.show()