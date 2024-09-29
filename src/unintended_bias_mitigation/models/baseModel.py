import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.evaluation.metrics import Metrics
from unintended_bias_mitigation.evaluation.nuancedMetrics import NuancedMetric
from unintended_bias_mitigation.data.preprocessor import ToxicityPreprocessor


warnings.filterwarnings('ignore')


class BaseModel:
    """Base model."""

    def __init__(self, model_params=None, training_params=None, dataset_params=None):
        self.model_dir = cfg.DEFAULT_MODEL_DIR
        self.embeddings_path = cfg.DEFAULT_EMBEDDINGS_PATH
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_params = training_params
        self.dataset_params = dataset_params
        self.model_params = model_params
        self.model_class = None
        self.model = None
        self.embedding_matrix = None
        self.dataset_preprocessor = None
        self.trainer = None
        self.model_trainer = None
        self.model_name = None
        if self.dataset_params is not None:
            self.model_folder = cfg.DATASET_IDENTITY[self.dataset_params['dataset_identifier']]
        self.get_model_class(self.model_params['model_class'])
        if self.model_params['model_name'] is not None:
            self.model_name = self.model_params['model_name']
            self.load_checkpoint()
        self.__init_preprocessor()
        if self.model_trainer is None:
            self.get_model_trainer()
        self.print_hparams()

    def __init_preprocessor(self):
        self.dataset_preprocessor = ToxicityPreprocessor(self.device, params=self.dataset_params)

    def set_hparams(self, training_params, dataset_params):
        if training_params:
            self.training_params.update(training_params)
        if dataset_params:
            self.dataset_params.update(dataset_params)

    def print_hparams(self):
        print('Hyper-parameters Model')
        print('---------------')
        for k, v in self.model_params.items():
            print('{}: {}'.format(k, v))
        print('')
        print('Hyper-parameters for Trainer')
        print('---------------')
        for k, v in self.training_params.items():
            print('{}: {}'.format(k, v))
        print('')

    def update_hparams(self, new_hparams):
        self.model_params.update(new_hparams)

    def get_model_name(self):
        return self.model_name

    def save_hparams(self, model_name):
        self.model_params['model_name'] = model_name
        hparams = {'model_params': self.model_params,
                   'training_params': self.training_params,
                   'dataset_params':  self.dataset_params}
        hparam_filename = os.path.join(cfg.DEFAULT_HPARAMS_DIR,
                                       f"{self.model_folder}/{self.model_name}_hparams.json")
        with open(hparam_filename, 'w') as f:
            json.dump(hparams, f, sort_keys=True)

    def load_checkpoint(self):
        """Load model given its name."""
        checkpoint = torch.load(os.path.join(self.model_dir, f"{self.model_folder}/{self.model_name}.pt"))
        hparam_filename = os.path.join(cfg.DEFAULT_HPARAMS_DIR,
                                       f"{self.model_folder}/{self.model_name}_hparams.json")
        with open(os.path.join(cfg.DEFAULT_HPARAMS_DIR, hparam_filename), 'r') as jsonfile:
            hparams = json.load(jsonfile)
            self.model_params = hparams['model_params']
            self.training_params = hparams['training_params']
            self.dataset_params = hparams['dataset_params']
        self.dataset_params.update({'use_bert': False})
        if self.model is None:
            self.model = self.model_class(self.model_params, use_feature=self.dataset_params['use_feature'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            print(self.model)
        if self.model_trainer is None:
            self.get_model_trainer()
        if self.trainer is None:
            self.trainer = self.model_trainer(self.model, self.device, self.training_params)
        print("Model loaded Successfully!!")

    def get_model_class(self, model_class):
        module = __import__(f"unintended_bias_mitigation.neuralNet.{model_class.lower()}", fromlist=[model_class])
        self.model_class = getattr(module, model_class)

    def get_model_trainer(self):
        if self.dataset_params['use_feature']:
            module = __import__("unintended_bias_mitigation.trainer.featured_trainer",
                                fromlist=["FeaturedTrainer"])
            self.model_trainer = getattr(module, "FeaturedTrainer")
        elif self.dataset_params['use_bert']:
            module = __import__("unintended_bias_mitigation.trainer.bertTrainer",
                                fromlist=["BERTTrainer"])
            self.model_trainer = getattr(module, "BERTTrainer")
        else:
            module = __import__("unintended_bias_mitigation.trainer.trainer",
                                fromlist=["ModelTrainer"])
            self.model_trainer = getattr(module, "ModelTrainer")

    def load_embeddings(self):
        """Loads word embeddings."""
        embeddings_index = {}
        with open(cfg.DEFAULT_EMBEDDINGS_PATH) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        self.embedding_matrix = torch.zeros((len(self.dataset_preprocessor.tokenizer.word_index) + 1,
                                             self.model_params['embedding_dim']))
        num_words_in_embedding = 0
        for word, i in self.dataset_preprocessor.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                num_words_in_embedding += 1
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = torch.from_numpy(embedding_vector)

    def train_model(self, train_dl, valid_dl):
        if self.training_params['use_embedding']:
            self.load_embeddings()
        self.model = self.model_class(self.model_params,
                                      embedding_matrix=self.embedding_matrix,
                                      use_feature=self.dataset_params['use_feature'])
        self.model.to(self.device)
        print("Model details: ")
        print(self.model)
        save_path = os.path.join(self.model_dir, f"{self.model_folder}/{self.model_name}.pt")
        if self.dataset_params['use_bert']:
            self.training_params['epochs'] = 4
        if self.trainer is None:
            self.trainer = self.model_trainer(self.model, self.device, self.training_params, save_path=save_path)
        self.trainer.train(train_dl, valid_dl)
        print('Model trained!')
        print(f'Best model saved to {save_path}')

    def build_model(self, model_name=None):
        """Trains the model."""
        if self.model_name is None and model_name is not None:
            self.model_name = model_name
            train_dl, valid_dl = self.dataset_preprocessor.prep_training_data()
            self.training_params.update({'n_training_samples': self.dataset_preprocessor.params['n_training_samples'],
                                         'n_validation_samples': self.dataset_preprocessor.params['n_validation_samples']})
            if not self.dataset_params['use_bert']:
                self.model_params.update({'vocab_size': self.dataset_preprocessor.params['vocab_size']})
            if self.dataset_params['use_feature']:
                self.model_params.update({'n_features': self.dataset_preprocessor.params['n_features']})
            print('Saving hyper-parameters...')
            self.save_hparams(model_name)
            print('Saved hyper-parameters')
            print('Training model...')
            self.train_model(train_dl, valid_dl)
        else:
            self.evaluate_model()
            self.evaluate_model(idenitity_phrases=True)

    def test_model(self, test_df, prediction_file, idenitity_phrases=False):
        test_dl = self.dataset_preprocessor.prep_test_data(test_df.copy(), idenitity_phrases=identity_phrases)
        self.training_params.update({'n_test_samples': self.dataset_preprocessor.params['n_test_samples']})
        labels, predictions, scores = self.trainer.predict_label(test_dl)
        if 'labels' not in test_df.columns:
            test_df['labels'] = labels.astype('int32')
        test_df[f"{self.model_name}_Predictions"] = predictions.tolist()
        test_df[f"{self.model_name}_Scores_0"] = scores[:, 0].tolist()
        test_df[f"{self.model_name}_Scores_1"] = scores[:, 1].tolist()
        test_df.to_csv(prediction_file, index=False)

    def evaluate_model(self, save_report=True, bias_analysis=True, idenitity_phrases=False):
        print('Evaluating model...')
        raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[self.dataset_params['dataset_identifier']])
        prediction_file = os.path.join(raw_folder_path, "test_predictions.csv")
        if idenitity_phrases:
            prediction_file = os.path.join(raw_folder_path, "identity_predictions.csv")
        test_file = self.dataset_params['test_file']
        if idenitity_phrases:
            test_file = "identity_phrase_templates"
        if not os.path.isfile(prediction_file):
            test_df = pd.read_csv(os.path.join(raw_folder_path, f"{test_file}.csv"))
        else:
            test_df = pd.read_csv(prediction_file)
        if f"{self.model_name}_Predictions" not in test_df.columns:
            self.test_model(test_df, prediction_file, idenitity_phrases)
        eval_filename = f"evaluation_{self.dataset_params['dataset_identifier']}"
        if idenitity_phrases:
            eval_filename = f"{eval_filename}_identity"
        if save_report:
            labels = test_df['labels']
            predictions = test_df[f"{self.model_name}_Predictions"]
            scores = test_df[f"{self.model_name}_Scores_1"]
            test_metric = Metrics(labels, predictions, scores)
            test_metric.save_evaluation_report(model_name=self.model_name,
                                               filename=eval_filename)
        if bias_analysis:
            bias_eval_file = f"bias_analysis_{self.dataset_params['dataset_identifier']}"
            if idenitity_phrases:
                bias_eval_file = f"{bias_eval_file}_identity"
            self.evaluate_bias(test_df, eval_filename, bias_eval_file)

    def evaluate_bias(self, test_df, eval_filename, bias_eval_file):
        identity_terms_path = os.path.join('../../data', 'identity_terms.txt')
        with open(identity_terms_path) as f:
            DEBIAS_TERMS = [term.strip() for term in f.readlines()]

        bias_metrics = NuancedMetric(data=test_df, subgroups=DEBIAS_TERMS, model_names=[self.model_name])
        bias_metrics.evaluate_model(eval_filename=eval_filename, bias_metrics_file=bias_eval_file)

