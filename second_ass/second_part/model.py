#from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel

class JointBERT(nn.Module):
    def __init__(self, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__()
        #self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # Load pretrained bert
        
        hid_size = self.bert.config.hidden_size

        self.intent_classifier = IntentClassifier(hid_size, self.num_intent_labels, 0.1)
        self.slot_classifier = SlotClassifier(hid_size, self.num_slot_labels, 0.1)

        # if args.use_crf:
        #     self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        slot_logits = slot_logits.permute(0,2,1)
        
        return slot_logits,intent_logits
   
    @property
    def device(self):
        return next(self.parameters()).device
 
class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    