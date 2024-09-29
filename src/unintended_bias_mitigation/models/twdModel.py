import os
import warnings
import numpy as np
import pandas as pd
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.evaluation.metrics import Metrics
from unintended_bias_mitigation.evaluation.nuancedMetrics import NuancedMetric

warnings.filterwarnings('ignore')


def reclassify(b_predictions, e_predictions, b_scores, e_scores):
    tox_boundary_label = cfg.BOUNDARY_LABELS['toxic']
    non_tox_boundary_label = cfg.BOUNDARY_LABELS['non_toxic']
    for idx in range(len(b_predictions)):
        if b_predictions[idx] == tox_boundary_label or b_predictions[idx] == non_tox_boundary_label:
            b_predictions[idx] = e_predictions[idx]
            b_scores[idx] = e_scores[idx]
    return b_predictions, b_scores


class TWDModel:
    def __init__(self, base_model, enhance_model=None, boundary_threshold=0.1):
        self.base_model = base_model
        self.enhance_model = enhance_model
        self.boundary_threshold = boundary_threshold

    def confidence_divider(self, scores, labels):
        toxic_count = 0
        non_toxic_count = 0
        for idx, score in enumerate(scores):
            if abs(score) <= self.boundary_threshold:
                if score > 0:
                    labels[idx] = cfg.BOUNDARY_LABELS['toxic']
                    toxic_count += 1
                else:
                    labels[idx] = cfg.BOUNDARY_LABELS['non_toxic']
                    non_toxic_count += 1
        print(f"for boundary threshold {self.boundary_threshold}")
        print(f"{(toxic_count + non_toxic_count)} {toxic_count} {non_toxic_count}")
        return labels

    def evaluate_model(self, enhance_model_name=None, idenitity_phrases=False):
        dataset_identifier = self.base_model.dataset_params['dataset_identifier']
        raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[dataset_identifier])
        test_file = "test_predictions"
        eval_file = f"evaluation_{dataset_identifier}_twd"
        bias_eval_file = f"bias_analysis_{dataset_identifier}_twd"
        if idenitity_phrases:
            test_file = "identity_predictions"
            eval_file = f"evaluation_{dataset_identifier}_identity_twd"
            bias_eval_file = f"bias_analysis_{dataset_identifier}_identity_twd"
        prediction_file = os.path.join(raw_folder_path, f"{test_file}.csv")
        base_model_name = self.base_model.model_params['model_name']
        if enhance_model_name is None:
            enhance_model_name = self.enhance_model.model_params['model_name']
        twd_model_name = f"{base_model_name}_{enhance_model_name}_{self.boundary_threshold}"
        if not os.path.isfile(prediction_file):
            self.base_model.evaluate_model(save_report=False, idenitity_phrases=idenitity_phrases)
            self.enhance_model.evaluate_model(save_report=False, idenitity_phrases=idenitity_phrases)
        test_df = pd.read_csv(prediction_file)
        if f"{base_model_name}_Predictions" not in test_df.columns:
            self.base_model.evaluate_model(save_report=False, idenitity_phrases=idenitity_phrases)
        if f"{enhance_model_name}_Predictions" not in test_df.columns:
            self.enhance_model.evaluate_model(save_report=False, idenitity_phrases=idenitity_phrases)
        labels = test_df['labels']
        b_predictions = test_df[f"{base_model_name}_Predictions"]
        e_predictions = test_df[f"{enhance_model_name}_Predictions"]
        b_scores_0 = test_df[f"{base_model_name}_Scores_0"]
        b_scores_1 = test_df[f"{base_model_name}_Scores_1"]
        b_scores = np.column_stack((b_scores_0, b_scores_1))
        e_scores_0 = test_df[f"{enhance_model_name}_Scores_0"]
        e_scores_1 = test_df[f"{enhance_model_name}_Scores_1"]
        e_scores = np.column_stack((e_scores_0, e_scores_1))
        confidence_interval = np.diff(b_scores)
        bnd_labels = self.confidence_divider(confidence_interval, b_predictions)
        predicted_labels_enhanced, scores_enhanced = reclassify(bnd_labels, e_predictions, b_scores, e_scores)
        test_df[f"{twd_model_name}_Predictions"] = predicted_labels_enhanced.tolist()
        test_df[f"{twd_model_name}_Scores_0"] = scores_enhanced[:, 0]
        test_df[f"{twd_model_name}_Scores_1"] = scores_enhanced[:, 1]
        test_df.to_csv(prediction_file, index=False)
        test_metric = Metrics(labels, predicted_labels_enhanced, scores_enhanced[:, 1])
        test_metric.save_evaluation_report(model_name=twd_model_name,
                                           filename=eval_file)
        identity_terms_path = os.path.join('../../data', 'identity_terms.txt')
        with open(identity_terms_path) as f:
            IDENTITY_TERMS = [term.strip() for term in f.readlines()]
        bias_metrics = NuancedMetric(data=test_df, subgroups=IDENTITY_TERMS,
                                     model_names=[twd_model_name])
        bias_metrics.evaluate_model(eval_filename=eval_file, bias_metrics_file=bias_eval_file)
        test_df = pd.read_csv(prediction_file)
        test_df = test_df.drop(columns=[f"{twd_model_name}_Predictions",
                                        f"{twd_model_name}_Scores_0",
                                        f"{twd_model_name}_Scores_1"])
        test_df.to_csv(prediction_file, index=False)
