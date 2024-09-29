import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class BERT(nn.Module):
    def __init__(self, hparams, embedding_matrix=None, use_feature=False):
        super().__init__()
        self.output_size = hparams['output_size']
        self.hidden_dim = hparams['hidden_dim']
        self.bert_layer = DistilBertForSequenceClassification.from_pretrained(hparams["model_checkpoint"],
                                                                              num_labels=self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 1)
        self.linear_aux_out = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(hparams['dropout_rate'])

    def forward(self, sentence, attention_mask=None, labels=None):
        bert_out = self.bert_layer(input_ids=sentence, attention_mask=attention_mask, labels=labels).logits
        result = self.linear_out(self.dropout(bert_out))
        aux_result = self.linear_aux_out(bert_out)
        out = torch.cat([result, aux_result], 1)
        return out
