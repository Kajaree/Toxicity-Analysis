import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.trainer.trainer import ModelTrainer


def check_null(input_seq):
    if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
        print('invalid input detected at iteration ', input_seq)
        return True
    return False


class FeaturedTrainer(ModelTrainer):
    def train_epoch(self, train_dl):
        model = self.model.train()
        losses = []
        correct_predictions = 0
        for input_seq, label, stat in train_dl:
            self.optimizer.zero_grad()
            model.zero_grad()
            input_seq = input_seq.to(self.device)
            label = label.to(self.device)
            stat = stat.to(self.device)
            output = model(input_seq, stat)
            loss = self.criterion(output, label.long())
            predicted_label = torch.argmax(output, dim=1)
            correct_predictions += predicted_label.eq(label).sum()
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
            for input_seq, label, stat in val_dl:
                input_seq = input_seq.to(self.device)
                label = label.view(-1).to(self.device)
                stat = stat.to(self.device)
                output = model(input_seq, stat)
                loss = self.criterion(output.squeeze(), label.long())
                predicted_label = torch.argmax(output, dim=1)
                correct_predictions += predicted_label.eq(label).sum()
                losses.append(loss.item())
        return correct_predictions / self.hparams['n_validation_samples'], np.mean(losses)

    def predict_label(self, test_dl):
        model = self.model.eval()
        batch_size = cfg.VALID_BATCH_SIZE
        y_score = np.zeros((self.hparams['n_test_samples'], 2))
        y_pred = np.zeros((self.hparams['n_test_samples']))
        y_true = np.zeros((self.hparams['n_test_samples']))
        with torch.no_grad():
            for i, [input_seq, label, stat] in enumerate(test_dl):
                input_seq = input_seq.to(self.device)
                stat = stat.to(self.device)
                output = model(input_seq, stat)
                predicted = torch.argmax(output, dim=1)
                y_score[i * batch_size:(i + 1) * batch_size] = F.softmax(output, dim=1).cpu().numpy()
                y_pred[i * batch_size:(i + 1) * batch_size] = predicted.detach().cpu().squeeze().numpy()
                y_true[i * batch_size:(i + 1) * batch_size] = label.squeeze().numpy()
        return y_true.ravel(), y_pred.ravel(), y_score
