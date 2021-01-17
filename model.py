import torch.nn as nn
from transformers import BertModel
import config


class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_NAME)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).values()
        dropout_output = self.dropout(pooler_output)
        output = self.out(dropout_output)
        return output