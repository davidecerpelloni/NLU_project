from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from conll import evaluate

def train_loop_old(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['slots'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def train_loop(data_loader, optimizer, criterion_slots, criterion_intents, model, device, clip=5):
    model.train()
    loss_array = []
    
    for batch in data_loader:
        optimizer.zero_grad()  # Zeroing the gradient

        input_ids = batch['utterance'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        slot_labels = batch['slots'].to(device)
        intent_labels = batch['intents'].to(device)

        # Forward pass
        slot_logits, intent_logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss_intent = criterion_intents(intent_logits, intent_labels)
        # Reshape logits and labels to ensure compatibility
        slot_logits_flat = slot_logits.reshape(-1, model.num_slot_labels)
        slot_labels_flat = slot_labels.reshape(-1)
        #loss_slot = criterion_slots(slot_logits_flat, slot_labels_flat)
        loss_slot = criterion_slots(slot_logits,slot_labels)
        loss = loss_intent + loss_slot  # Joint loss

        loss_array.append(loss.item())
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
    return loss_array

def eval_loop_old(data, criterion_slots, criterion_intents, model, lang): #eval not taken padding
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class name
            out_intents = [lang.id2intent[x]
                            for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] 
                            for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def eval_loop_lessold(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            print(sample)

            # Move tensors to the same device as the model
            sample = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            
            # Forward pass
            slots, intents = model(sample['utterance'], sample['attention_mask'])
            
            # Compute losses
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots.reshape(-1, slots.size(-1)), sample['slots'].reshape(-1))
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=-1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['attention_mask'][id_seq].sum().item()
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def eval_loop2(data_loader, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data_loader:
            # Move tensors to the same device as the model
            sample = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            
            # Forward pass
            slots, intents = model(sample['utterance'], sample['attention_mask'])

            # Compute losses
            print(f"intents shape: {intents.shape}, target intents shape: {sample['intents'].shape}")
            print(f"slots shape: {slots.shape}, target slots shape: {sample['slots'].shape}")
            
            loss_intent = criterion_intents(intents, sample['intents'])            
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=-1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['attention_mask'][id_seq].sum().item()  # Length of the actual content
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = [(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)]
                hyp_slots.append(tmp_seq)

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            # print(sample)
            # exit()
            # Move tensors to the same device as the model
            sample = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            # sample = {'utterance': sample['utterance'].to(model.device),
            #       'attention_mask': sample['attention_mask'].to(model.device),
            #       'slots' : sample['slots'].to(model.device),
            #       'intents' : sample['intents'].to(model.device)}
            
            # Forward pass
            slots, intents = model(sample['utterance'], sample['attention_mask'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['attention_mask'][id_seq].sum().item()  # Length of the actual content
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                utterance = [lang.id2word[elem] for elem in utt_ids]

                # utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                #utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                tmp_ref_slots = []
                tmp_tmp_seq = []
                #ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                for id_el, elem in enumerate(to_decode):
                    tmp_tmp_seq.append((utterance[id_el], lang.id2slot[elem]))

                # remove pad tokens
                ok_ref_slots = []
                ok_tmp_seq = []
                for i in range(len(tmp_ref_slots[0])):
                    if tmp_ref_slots[0][i][1] != 'pad':
                        ok_ref_slots.append(tmp_ref_slots[0][i])
                        ok_tmp_seq.append(tmp_tmp_seq[i])
                        
                ref_slots.append(ok_ref_slots)
                hyp_slots.append(ok_tmp_seq)
    print(ref_slots[1])
    print(hyp_slots[1])
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

