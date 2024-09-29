import re
import os
import numpy as np
import pandas as pd
from IPython.display import display
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.evaluation.metrics import Metrics


def compute_auc(y_true, y_pred, scores):
    test_metric = Metrics(y_true, y_pred, scores)
    try:
        return test_metric.auroc_score()
    except ValueError:
        return np.nan


class NuancedMetric:
    def __init__(self, data, subgroups, model_names, power=-5, overall_model_weight=0.25):
        self.dataset = data
        self.subgroups = subgroups
        self.model_names = model_names
        self.power = power
        self.overall_model_weight = overall_model_weight
        self.__convert_labels()
        self.add_subgroup_columns_from_text()

    def __convert_labels(self):
        for subgroup in self.subgroups:
            if subgroup in self.dataset.columns:
                self.dataset[subgroup] = self.dataset[subgroup].fillna(0)
                self.dataset[subgroup] = (self.dataset[subgroup].values > 0.5)

    def add_subgroup_columns_from_text(self):
        """Adds a boolean column for each subgroup to the data frame.
        New column contains True if the text contains that subgroup term.
        """
        for term in self.subgroups:
            if term not in self.dataset.columns:
                self.dataset[term] = self.dataset[cfg.TEXT_COLUMN].apply(
                    lambda x: bool(re.search('\\b' + term + '\\b', x, flags=re.UNICODE | re.IGNORECASE)))
                # print(term, self.dataset[term])

    def compute_subgroup_auc(self, subgroup, model_name):
        subgroup_examples = self.dataset[self.dataset[subgroup]]
        return compute_auc(subgroup_examples['labels'],
                           subgroup_examples[f"{model_name}_Predictions"],
                           subgroup_examples[f"{model_name}_Scores_1"])

    def compute_bpsn_auc(self, subgroup, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = self.dataset[self.dataset[subgroup] & ~self.dataset['labels']]
        non_subgroup_positive_examples = self.dataset[~self.dataset[subgroup] & self.dataset['labels']]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return compute_auc(examples['labels'], examples[f"{model_name}_Predictions"], examples[f"{model_name}_Scores_1"])

    def compute_bnsp_auc(self, subgroup, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = self.dataset[self.dataset[subgroup] & self.dataset['labels']]
        non_subgroup_negative_examples = self.dataset[~self.dataset[subgroup] & ~self.dataset['labels']]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return compute_auc(examples['labels'], examples[f"{model_name}_Predictions"], examples[f"{model_name}_Scores_1"])

    def compute_bias_metrics(self, model_name):
        evaluation_report = []
        for subgroup in self.subgroups:
            subgroup_size = len(self.dataset[self.dataset[subgroup]])
            if subgroup_size >= 60:
                report = {
                    'model name': model_name,
                    'subgroup': subgroup,
                    'subgroup_size': subgroup_size,
                    'SUBGROUP_AUC': self.compute_subgroup_auc(subgroup, model_name),
                    'BPSN_AUC': self.compute_bpsn_auc(subgroup, model_name),
                    'BNSP_AUC': self.compute_bnsp_auc(subgroup, model_name)
                }
                evaluation_report.append(report)
        return pd.DataFrame(evaluation_report).sort_values('SUBGROUP_AUC', ascending=True)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, model_name, bias_metrics=None):
        y_true = self.dataset['labels']
        y_pred = self.dataset[f"{model_name}_Predictions"]
        y_score = self.dataset[f"{model_name}_Scores_1"]
        if bias_metrics is None:
            bias_metrics = self.compute_bias_metrics(model_name)
        bias_score = np.average([
            self._power_mean(bias_metrics['SUBGROUP_AUC']),
            self._power_mean(bias_metrics['BPSN_AUC']),
            self._power_mean(bias_metrics['BNSP_AUC'])
        ])
        overall_score = self.overall_model_weight * compute_auc(y_true, y_pred, y_score)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score

    def evaluate_model(self, eval_filename, bias_metrics_file):
        result = None
        general_eval_report_file_name = os.path.join(cfg.DEFAULT_EVALS_DIR, f"{eval_filename}.csv")
        general_eval_report = pd.read_csv(general_eval_report_file_name, encoding='utf-8')
        if 'overall_bias_score' not in general_eval_report['metrics'].values:
            general_eval_report.loc[len(general_eval_report)] = None
            general_eval_report.loc[len(general_eval_report) - 1, 'metrics'] = 'overall_bias_score'
        filename = os.path.join(cfg.DEFAULT_EVALS_DIR, f"{bias_metrics_file}.csv")
        if os.path.exists(filename):
            result = pd.read_csv(filename, encoding='utf-8')
        for model in self.model_names:
            print('')
            metrics_df = self.compute_bias_metrics(model)
            general_eval_report.loc[len(general_eval_report) - 1, model] = self.get_final_metric(model, metrics_df)
            if result is None:
                result = metrics_df
            else:
                result = pd.concat([result, metrics_df])
        display(result)
        general_eval_report.to_csv(general_eval_report_file_name, encoding='utf-8', index=False)
        result.to_csv(filename, encoding='utf-8', index=False)
        print(f'Bias evaluation report is saved to {bias_metrics_file}')





