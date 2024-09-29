import torch.nn as nn
import torch
import math


class CNN(nn.Module):
    def __init__(self, hparams, embedding_matrix=None, use_feature=False):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(self.hparams['vocab_size'], self.hparams['embedding_dim'])
        if embedding_matrix:
            self.embedding_dim = embedding_matrix.shape[1]
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv1d(in_channels=self.hparams['embedding_dim'],
                               out_channels=self.hparams['cnn_filter_size'],
                               kernel_size=self.hparams['cnn_kernel_size'],
                               padding=math.floor(self.hparams['cnn_kernel_size']/2))
        self.pool = nn.MaxPool1d(self.hparams['cnn_pooling_size'],
                                 padding=math.floor(self.hparams['cnn_kernel_size']/2))
        self.conv2 = nn.Conv1d(in_channels=self.hparams['cnn_filter_size'],
                               out_channels=self.hparams['cnn_filter_size'],
                               kernel_size=self.hparams['cnn_kernel_size'],
                               padding=math.floor(self.hparams['cnn_kernel_size'] / 2))
        self.conv3 = nn.Conv1d(in_channels=self.hparams['cnn_filter_size'],
                               out_channels=self.hparams['cnn_filter_size'],
                               kernel_size=self.hparams['cnn_kernel_size'],
                               padding=math.floor(self.hparams['cnn_kernel_size'] / 2))
        self.dropout = nn.Dropout(self.hparams['dropout_rate'])
        self.fc = nn.Linear(self.hparams['hidden_dim'], self.hparams['output_size'])
        if use_feature:
            self.fc_stat = nn.Linear(self.hparams['n_features'], 64)
            self.fc = nn.Linear(self.hparams['hidden_dim'] + 64, self.hparams['output_size'])
        self.activation = nn.ReLU()

    def forward(self, sentence, features=None):
        embed_input = self.embedding(sentence.long())
        embed_input = embed_input.permute(0, 2, 1)
        out = self.pool(self.activation(self.conv1(embed_input)))
        out = self.pool(self.activation(self.conv2(out)))
        out = self.pool(self.activation(self.conv3(out)))
        out = torch.flatten(out, 1)  # flatten all dimensions except batch
        output = self.dropout(out)
        if features is not None:
            stat_out = self.fc_stat(features)
            output = torch.cat((output, stat_out), 1)
        output = self.activation(self.fc(output))
        return output
