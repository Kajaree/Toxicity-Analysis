import torch
import torch.nn as nn
import torch.nn.functional as F


class BILSTM(nn.Module):
    def __init__(self, hparams, embedding_matrix=None, use_feature=False):
        super().__init__()
        self.output_size = hparams['output_size']
        self.n_layers = hparams['n_layers']
        self.hidden_dim = hparams['hidden_dim']
        self.embedding_dim = hparams['embedding_dim']
        # embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(hparams['vocab_size'], self.embedding_dim)
        if embedding_matrix:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(hparams['dropout_rate'])

        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, bidirectional=True, batch_first=True)
        linear_output_dim = 4 * self.hidden_dim
        if use_feature:
            linear_output_dim += 64
            self.fc_stat = nn.Linear(hparams['num_features'], 64)

        self.fc = nn.Linear(linear_output_dim, linear_output_dim)
        self.fc_out = nn.Linear(linear_output_dim, 1)
        self.fc_aux_out = nn.Linear(linear_output_dim, 1)

    def forward(self, comments, features=None):
        h_embedding = self.embedding(comments.long())
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        # stat feature out
        if features is not None:
            stat_out = self.fc_stat(features)
            h_conc = torch.cat((max_pool, avg_pool, stat_out), 1)
        else:
            h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear = F.relu(self.fc(h_conc))
        hidden = h_conc + h_conc_linear
        result = self.fc_out(hidden)
        aux_result = self.fc_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        return out
