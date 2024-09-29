import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.trainer.trainer import ModelTrainer


class BERTTrainer(ModelTrainer):
    def __init__(self, model, device, hparams, save_path=None):
        super().__init__(model, device, hparams, save_path=save_path)
        self.__initialize_optimizer()

    def __initialize_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=self.hparams['learning_rate'], warmup=.1)

    def train_epoch(self, train_dl):
        model = self.model.train()
        losses = []
        correct_predictions = 0
        for input_seq, input_mask, label_seq in train_dl:
            input_seq = input_seq.to(self.device)
            label_seq = label_seq.to(self.device)
            input_mask = input_mask.to(self.device)
            self.optimizer.zero_grad()
            output = model(input_seq, attention_mask=input_mask, labels=None)
            loss = self.criterion(output, label_seq.long())
            predicted_label = torch.argmax(output, dim=1)
            correct_predictions += predicted_label.eq(label_seq).sum()
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.hparams['clip'])
            self.optimizer.step()
        return correct_predictions / self.hparams['n_training_samples'], np.mean(losses)

    def eval_model(self, val_dl):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for input_seq, input_mask, label_seq in val_dl:
                input_seq = input_seq.to(self.device)
                label_seq = label_seq.to(self.device)
                input_mask = input_mask.to(self.device)
                self.optimizer.zero_grad()
                output = model(input_seq, attention_mask=input_mask, labels=None)
                loss = self.criterion(output.squeeze(), label_seq.long())
                predicted_label = torch.argmax(output, dim=1)
                correct_predictions += predicted_label.eq(label_seq).sum()
                losses.append(loss.item())
        return correct_predictions / self.hparams['n_validation_samples'], np.mean(losses)

    def predict_label(self, test_dl):
        model = self.model.eval()
        batch_size = cfg.VALID_BATCH_SIZE
        y_score = np.zeros((self.hparams['n_test_samples'], 2))
        y_pred = np.zeros((self.hparams['n_test_samples']))
        y_true = np.zeros((self.hparams['n_test_samples']))
        with torch.no_grad():
            for i, [input_seq, input_mask, label] in enumerate(test_dl):
                input_seq = input_seq.to(self.device)
                input_mask = input_mask.to(self.device)
                self.optimizer.zero_grad()
                output = model(input_seq, attention_mask=input_mask, labels=None)
                predicted = torch.argmax(output, dim=1)
                y_score[i * batch_size:(i + 1) * batch_size] = F.softmax(output, dim=1).cpu().numpy()
                y_pred[i * batch_size:(i + 1) * batch_size] = predicted.detach().cpu().squeeze().numpy()
                y_true[i * batch_size:(i + 1) * batch_size] = label.squeeze().numpy()
        return y_true.ravel(), y_pred.ravel(), y_score