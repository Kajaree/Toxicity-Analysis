import os
import pandas as pd
from IPython.display import display
import unintended_bias_mitigation.utils.config as cfg
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


class Metrics:
    def __init__(self, labels, predictions, score=None, pos_label=1):
        self.y_true = labels
        self.y_pred = predictions
        self.pos_label = pos_label
        self.scores = score

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='macro')

    def aupr_score(self):
        return average_precision_score(self.y_true, self.y_pred, average='macro')

    def auroc_score(self):
        if self.scores is None:
            return roc_auc_score(self.y_true, self.y_pred, average='macro')
        else:
            return roc_auc_score(self.y_true, self.scores, average='macro')

    def save_evaluation_report(self, model_name, filename):
        print(f'Evaluation report for model {model_name}')
        eval_metrics = cfg.DEFAULT_EVAL_METRICS
        eval_reports = [0] * len(eval_metrics)
        for i in range(len(eval_metrics) - 1):
            eval_reports[i] = getattr(self, eval_metrics[i])()
        file_name = os.path.join(cfg.DEFAULT_EVALS_DIR, f"{filename}.csv")
        if os.path.exists(file_name):
            eval_report = pd.read_csv(file_name, encoding='utf-8')
        else:
            eval_report = pd.DataFrame()
            eval_report['metrics'] = eval_metrics
        eval_report[model_name] = eval_reports
        display(eval_report)
        eval_report.to_csv(file_name, encoding='utf-8', index=False)
        print(f'Evaluation report is saved to {filename}')
