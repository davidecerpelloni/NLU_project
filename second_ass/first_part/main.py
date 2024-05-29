import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from functions import *


device = 'cpu' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

# tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
# test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

print('Train samples:', len(tmp_train_raw))
print('Test samples:', len(test_raw))

print(tmp_train_raw[0])

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
print({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())}) #round(v/len(y_train),3)*100 v/len, tiene i primi 3 decimali, moltiplica per 100
print('Dev:'),
print({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})  #print(Counter(y_train).items())   dict_items([('airfare', 381), ('flight', 3299),...])
print('Test:')
print({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
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

lang = Lang(words, intents, slots, cutoff=0)

# Create our datasets
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)
print(train_dataset[0])

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


hid_size = 250
emb_size = 300

lr = 0.0004 # learning rate
clip = 5 # Clip the gradient

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

n_epochs = 200
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
for x in tqdm(range(1,n_epochs)):
    loss = train_loop(train_loader, optimizer, criterion_slots,
                      criterion_intents, model, clip=clip)
    if x % 5 == 0: # We check the performance every 5 epochs
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                      criterion_intents, model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())

        f1 = results_dev['total']['f']
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
plt.show()