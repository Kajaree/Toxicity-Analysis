import torch
import torch.nn as nn


class LSTM(torch.nn.Module):
    def __init__(self, hparams, embedding_matrix=None, use_feature=False):
        super().__init__()
        self.output_size = hparams['output_size']
        self.n_layers = hparams['n_layers']
        self.hidden_dim = hparams['hidden_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.embedding = nn.Embedding(hparams['vocab_size'], self.embedding_dim)
        if embedding_matrix is not None:
            self.embedding_dim = embedding_matrix.shape[1]
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(hparams['dropout_rate'])
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        if use_feature:
            self.fc = nn.Linear(self.hidden_dim + 64, self.output_size)
            self.fc_stat = nn.Linear(hparams['n_features'], 64)
        self.relu = nn.ReLU()

    def forward(self, sentence, features=None):
        embed_input = self.embedding(sentence.long())
        _, (ht, _) = self.lstm(self.dropout(embed_input))
        output = ht[-1]
        if features is not None:
            stat_out = self.dropout(self.fc_stat(features))
            output = torch.cat((output, stat_out), 1)
        output = self.relu(output)
        out = self.fc(output)
        return out
