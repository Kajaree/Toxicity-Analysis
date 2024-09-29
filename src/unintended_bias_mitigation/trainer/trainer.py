import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import unintended_bias_mitigation.utils.config as cfg


def check_null(input_seq):
    if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
        print('invalid input detected at iteration ', input_seq)
        return True
    return False


class ModelTrainer:
    def __init__(self, model, device, hparams, save_path=None):
        self.model = model
        self.hparams = hparams
        self.epochs = hparams['epochs']
        self.device = device
        self.parameters = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = None
        self.criterion = getattr(nn, self.hparams['criterion'])().to(device)
        if save_path is not None:
            self.model_save_path = save_path
            self.history = defaultdict(list)
            self.__initialize_history()
        self.set_optimizer()

    def __initialize_history(self):
        self.history['learning_rate'] = [self.hparams['learning_rate']]
        self.history['optimizer'] = [self.optimizer]
        self.history['criterion'] = [self.criterion]
        self.history['epochs'] = [self.epochs]

    def set_optimizer(self):
        self.optimizer = getattr(torch.optim, self.hparams['optimizer'])(self.parameters,
                                                                         lr=self.hparams['learning_rate'],
                                                                         weight_decay=0.003)

    def train_epoch(self, train_dl):
        self.model.train()
        losses = []
        correct_predictions = 0
        for input_seq, label_seq in train_dl:
            self.optimizer.zero_grad()
            self.model.zero_grad()
            input_seq = input_seq.to(self.device)
            label_seq = label_seq.to(self.device)
            output = self.model(input_seq)
            loss = self.criterion(output, label_seq.long())
            predicted_label = torch.argmax(output, dim=1)
            correct_predictions += predicted_label.eq(label_seq).sum()
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams['clip'])
            self.optimizer.step()
        return correct_predictions / self.hparams['n_training_samples'], np.mean(losses)

    def eval_model(self, val_dl):
        model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for input_seq, label in val_dl:
                input_seq = input_seq.to(self.device)
                label_seq = label.view(-1).to(self.device)
                output = model(input_seq)
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
            for i, [input_seq, label] in enumerate(test_dl):
                input_seq = input_seq.to(self.device)
                output = F.softmax(model(input_seq), dim=1)
                predicted = torch.argmax(output, dim=1)
                y_score[i*batch_size:(i+1)*batch_size] = output.cpu().numpy()
                y_pred[i*batch_size:(i+1)*batch_size] = predicted.detach().cpu().squeeze().numpy()
                y_true[i * batch_size:(i + 1) * batch_size] = label.squeeze().numpy()
        return y_true.ravel(), y_pred.ravel(), y_score

    def train(self, train_dl, val_dl):
        torch.cuda.empty_cache()
        best_accuracy = 0
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.epochs):
                train_acc, train_loss = self.train_epoch(train_dl)
                val_acc, val_loss = self.eval_model(val_dl)
                self.__print_training_info(epoch, train_acc, train_loss, val_acc, val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['train_loss'].append(train_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_loss'].append(val_loss)
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    self.__save_model(epoch + 1, train_loss, train_acc)

    def __print_training_info(self, epoch, train_acc, train_loss, val_acc, val_loss):
        print(f'Epoch {epoch + 1}/{self.epochs}')
        print('-' * 10)
        print(f'Training   loss {train_loss} accuracy {train_acc}')
        print(f'Validation loss {val_loss} accuracy {val_acc}')
        print()

    def __save_model(self, epoch, loss, acc):
        torch.save({
            'epoch': epoch,
            'learning_rate': self.hparams['learning_rate'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_history': self.history,
            'loss': loss,
            'accuracy': acc
        }, self.model_save_path)
