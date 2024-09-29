import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import unintended_bias_mitigation.utils.config as cfg


model_dir = cfg.DEFAULT_MODEL_DIR
plot_dir = cfg.DEFAULT_PLOTS_DIR
data_dir = '../../data/'


def load_checkpoint(dataset_id, model_name):
    model_folder = cfg.DATASET_IDENTITY[dataset_id]
    checkpoint = torch.load(os.path.join(model_dir, f"{model_folder}/{model_name}.pt"))
    lr = checkpoint['learning_rate']
    history = checkpoint['model_history']
    train_acc = history['train_acc']
    train_loss = history['train_loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    del checkpoint
    del history
    title = f'Learning rate: {lr}'
    plot_file_name = os.path.join(plot_dir, f"{model_folder}/{model_name}_accuracy_compare.pdf")
    plot_train_valid(train_acc, val_acc, plot_file_name, title)
    plot_file_name = os.path.join(plot_dir, f"{model_folder}/{model_name}_loss_compare.pdf")
    plot_train_valid(train_loss, val_loss, plot_file_name, title, metric='Loss')


def plot_train_valid(training_data, valid_data, filename, title, metric='Accuracy'):
    fig = plt.figure()
    plt.plot(training_data, label=f"Training {metric}")
    plt.plot(valid_data, label=f"Validation {metric}")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_performance_twd(auc, bias_auc, aucs, bias_aucs, filename=None):
    fig = plt.figure()
    auc = [auc] * 20
    bias_auc = [bias_auc] * 20
    x = np.linspace(0.001, 0.5, num=20)
    plt.plot(x, aucs, 'g', label=f"AUC")
    plt.plot(x, bias_aucs, 'r', label=f"BIAS AUC")
    plt.plot(x, auc, 'g--')
    plt.plot(x, bias_auc, 'r--')
    plt.xlabel('Thresholds', fontsize=20)
    plt.ylabel('AUCs', fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close(fig)


if __name__ == '__main__':
    dataset_id = 'toxicity'
    base_models = ['LSTM_1', 'LSTM_2', 'LSTM_3', 'LSTM_4', 'LSTM_5']
    # for base_model in base_models:
    #     load_checkpoint(dataset_id, base_model)
    # enhance_models = ['lr', 'mnb']
    # model_folder = cfg.DATASET_IDENTITY[dataset_id]
    # # metric_idx = [0, 3, 4]
    # thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    #               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]
    filename = f"evaluation_{dataset_id}.csv"
    file_name = os.path.join(cfg.DEFAULT_EVALS_DIR, filename)
    eval_report = pd.read_csv(file_name, encoding='utf-8')
    display(eval_report[base_models])
    # # twd_eval_file = os.path.join(cfg.DEFAULT_EVALS_DIR, f"evaluation_{dataset_id}_identity_twd.csv")
    # # eval_report_twd = pd.read_csv(twd_eval_file, encoding='utf-8')
    # # # display(eval_report[base_models])
    # # for base_model in base_models:
    # #     auc = eval_report[base_model].iloc[3]
    # #     bias_auc = eval_report[base_model].iloc[4]
    # #     # print(auc, bias_auc)
    # #     for enhance_model in enhance_models:
    # #         model_name = f"{base_model}_{enhance_model}"
    # #         plot_file_name = os.path.join(plot_dir, f"{model_folder}/twd_performance/{model_name}_identity.pdf")
    # #         cols = [f"{model_name}_{threshold}" for threshold in thresholds]
    # #
    # #         aucs = eval_report_twd[cols].iloc[3].tolist()
    # #         bias_aucs = eval_report_twd[cols].iloc[4].tolist()
    # #         # max_bias_auc = max(bias_aucs)
    # #         # max_bias_auc_idx = bias_aucs.index(max_bias_auc)
    # #         # auc_at = aucs[max_bias_auc_idx]
    # #         # th = thresholds[max_bias_auc_idx]
    # #         # print(model_name, th, auc_at, round(auc_at - auc, 4), max_bias_auc, round(max_bias_auc - bias_auc, 4))
    # #         plot_performance_twd(auc, bias_auc, aucs, bias_aucs, plot_file_name)
    #
    # # print(eval_report.columns)
    # # eval_report.to_csv(file_name, encoding='utf-8', index=False)
    # # display(eval_report[['metrics', 'lr_augmented', 'mnb_augmented']])
    # # display(eval_report[['metrics', 'CNN_1_augmented', 'CNN_2_augmented']])
    # # display(eval_report[['CNN_3_augmented', 'CNN_4_augmented', 'CNN_5_augmented']])
    # display(eval_report[['metrics', 'BERT_1', 'BERT_2', 'BERT_3']])
    # display(eval_report[['metrics', 'LSTM_1_augmented', 'LSTM_2_augmented']])
    # display(eval_report[['LSTM_3_augmented', 'LSTM_4_augmented', 'LSTM_5_augmented']])
    # display(eval_report[['metrics', 'BILSTM_1_augmented', 'BILSTM_2_augmented']])
    # display(eval_report[['BILSTM_3_augmented', 'BILSTM_4_augmented', 'BILSTM_5_augmented']])
    # display(eval_report[enhance_models])
    # display(eval_report[['metrics', 'BILSTM_1', 'BILSTM_2', 'BILSTM_3', 'BILSTM_4', 'BILSTM_5']])
    # display(eval_report[['metrics', 'LSTM_1', 'LSTM_2', 'LSTM_3', 'LSTM_4', 'LSTM_5']])
    # display(eval_report[['metrics', 'CNN_1', 'CNN_2', 'CNN_3', 'CNN_4', 'CNN_5']])
    # display(eval_report[['metrics', 'BERT_1_augmented', 'BERT_2_augmented', 'BERT_3_augmented' ]])
