import os
import json
import torch
import string
import pickle
import pandas as pd
from joblib import dump, load
from sklearn.utils import resample
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer
from keras.preprocessing.text import Tokenizer
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import unintended_bias_mitigation.utils.config as cfg
from keras.preprocessing.sequence import pad_sequences
from unintended_bias_mitigation.data.cleaner import clean_data


def prep_data_loader(sequence, labels, batch_size, stats=None, masks=None):
    sequence = torch.tensor(sequence, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    if stats is not None:
        stats = torch.tensor(stats, dtype=torch.float32)
        data_set = torch.utils.data.TensorDataset(sequence, labels, stats)
    elif masks is not None:
        masks = torch.tensor(masks)
        data_set = torch.utils.data.TensorDataset(sequence, masks, labels)
    else:
        data_set = torch.utils.data.TensorDataset(sequence, labels)
    return DataLoader(data_set, shuffle=False, batch_size=batch_size)


def prep_features(data):
    data['word_count'] = data['comment_text'].apply(lambda x: len(x.split()))
    data['char_count'] = data['comment_text'].apply(lambda x: len(x.replace(" ", "")))
    data['word_density'] = data['word_count'] / (data['char_count'] + 1)
    data['punc_count'] = data['comment_text'].apply(lambda x: len([a for a in x if a in string.punctuation]))
    data['total_length'] = data['comment_text'].apply(len)
    data['capitals'] = data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    data['caps_vs_length'] = data.apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)
    data['num_exclamation_marks'] = data['comment_text'].apply(lambda x: x.count('!'))
    data['num_question_marks'] = data['comment_text'].apply(lambda x: x.count('?'))
    data['num_punctuation'] = data['comment_text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    data['num_symbols'] = data['comment_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    data['num_unique_words'] = data['comment_text'].apply(lambda x: len(set(w for w in x.split())))
    data['words_vs_unique'] = data['num_unique_words'] / data['word_count']
    data["word_unique_percent"] = data["num_unique_words"] * 100 / data['word_count']
    return data


def convert_labels(data, target_columns):
    data[target_columns] = (data[target_columns].values > 0.5).astype(int)
    target_columns.append(cfg.TEXT_COLUMN)
    return data[target_columns]


def balanced_sample(dataset_df):
    dataset_df['labels'] = dataset_df[list(dataset_df.columns)].sum(axis=1)
    dataset_df['labels'] = dataset_df['labels'].apply(lambda x: 1 if x > 1 else x)
    df_toxic = dataset_df[dataset_df['labels'] == 1]
    df_non_toxic = dataset_df[dataset_df['labels'] == 0]
    df_toxic_sampled = resample(df_toxic, replace=True, n_samples=df_toxic.shape[0], random_state=cfg.SEED)
    df_non_toxic_sampled = resample(df_non_toxic, replace=False, n_samples=cfg.SAMPLE_SIZE, random_state=cfg.SEED)
    df = pd.concat([df_non_toxic_sampled, df_toxic_sampled])
    return df.sample(frac=1, random_state=cfg.SEED)


def read_data(filepath):
    data = pd.read_csv(filepath)
    return data


class ToxicityPreprocessor:
    def __init__(self, device, params=None, split_name='train'):
        self.split_name = split_name
        self.tokenizer = None
        self.params = cfg.DEFAULT_DATSET_PARAMS.copy()
        self.features = None
        self.feature_normalizer = None
        self.device = device
        if params:
            self.update_params(params)
        filename = f"{self.params['dataset_identifier']}_tokenizer"
        if self.params['use_bert']:
            self.tokenizer_filename = os.path.join(cfg.DEFAULT_TOKENIZER_DIR, f"{filename}_bert.json")
        else:
            self.tokenizer_filename = os.path.join(cfg.DEFAULT_TOKENIZER_DIR, f"{filename}.pkl")

        self.normalizer_filename = os.path.join(cfg.DEFAULT_NORMALIZER_DIR,
                                                f"{self.params['dataset_identifier']}_normalizer.joblib")
        self.print_params()

    def print_params(self):
        print('data parameters')
        print('---------------')
        for k, v in self.params.items():
            print('{}: {}'.format(k, v))
        print('')

    def update_params(self, new_params):
        self.params.update(new_params)

    def initialize_tokenizer(self, data):
        if self.params['use_bert']:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        else:
            data = clean_data(data)
            if not os.path.exists(self.tokenizer_filename):
                print('Fitting tokenizer...')
                self.fit_and_save_tokenizer(data[cfg.TEXT_COLUMN])
                print('Tokenizer fitted!')
            else:
                self.tokenizer = pickle.load(open(self.tokenizer_filename, 'rb'))
            self.update_params({'vocab_size': len(self.tokenizer.word_index.keys()) + 1})

    def initialize_normalizer(self, data):
        if not os.path.exists(self.normalizer_filename):
            print('Fitting normalizer...')
            self.feature_normalizer = Normalizer()
            self.feature_normalizer.fit(data[self.features])
            dump(self.feature_normalizer, self.normalizer_filename)
            print('Tokenizer fitted!')
        else:
            self.feature_normalizer = load(self.normalizer_filename)

    def fit_and_save_tokenizer(self, texts):
        """Fits tokenizer on texts and pickles the tokenizer state."""
        self.tokenizer = Tokenizer(num_words=cfg.MAX_NUM_WORDS)
        self.tokenizer.fit_on_texts(texts)
        pickle.dump(self.tokenizer, open(self.tokenizer_filename, 'wb'))

    def fit_and_save_bert_tokenizer(self, texts):
        """Fits tokenizer on texts and pickles the tokenizer state."""

        input_ids = []
        attention_masks = []
        for text in texts:
            encoded_sent = self.tokenizer.encode_plus(
                text=text,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=cfg.MAX_SENTENCE_LENGTH,  # Max length to truncate/pad
                padding='max_length',  # Pad sentence to max length
                truncation=True,
                return_attention_mask=True  # Return attention mask
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        return input_ids, attention_masks

    def prep_text(self, texts):
        """Turns text into into padded sequences.
            The tokenizer must be initialized before calling this method.
            Args:
              texts: Sequence of text strings.
            Returns:
              A tokenized and padded text sequence as a model input.
        """
        if self.params['use_bert']:
            input_ids, attention_masks = self.fit_and_save_bert_tokenizer(texts)
            return [input_ids, attention_masks]
        else:
            text_sequences = self.tokenizer.texts_to_sequences(texts)
            return pad_sequences(text_sequences, maxlen=cfg.MAX_SENTENCE_LENGTH)

    def prep_training_data(self):
        raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[self.params['dataset_identifier']])
        file_name = os.path.join(raw_folder_path, f"{self.params['train_file']}.csv")
        data = read_data(file_name)
        print('Preparing training data...')
        if self.params['dataset_identifier'] == 'unintended_bias':
            data = data.rename(columns={"target": 'toxicity'})
            target_columns = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
            data = convert_labels(data, target_columns)
            data = balanced_sample(data)
        train = data.copy()
        if self.params['use_feature']:
            train = prep_features(train)
            self.features = list(set(train.columns) - set(data.columns))
            self.update_params({'n_features': len(self.features)})
            train[self.features] = train[self.features].fillna(0)
            self.initialize_normalizer(train)
            train[self.features] = self.feature_normalizer.transform(train[self.features])
        self.initialize_tokenizer(train)
        train, valid = train_test_split(train, train_size=self.params['split_ratio'], random_state=cfg.SEED)
        train_x, train_y = self.prepare_data(train)
        val_x, val_y = self.prepare_data(valid)
        self.update_params({'n_training_samples': len(train_y),
                            'n_validation_samples': len(val_y)})
        if self.features is not None:
            train_stat = train[self.features].values
            valid_stat = valid[self.features].values
            train_dl = prep_data_loader(train_x, train_y,
                                        batch_size=cfg.TRAIN_BATCH_SIZE, stats=train_stat)
            valid_dl = prep_data_loader(val_x, val_y,
                                        batch_size=cfg.VALID_BATCH_SIZE, stats=valid_stat)
        elif self.params['use_bert']:
            train_dl = prep_data_loader(train_x[0], train_y, batch_size=cfg.TRAIN_BATCH_SIZE, masks=train_x[1])
            valid_dl = prep_data_loader(val_x[0], val_y, batch_size=cfg.VALID_BATCH_SIZE, masks=val_x[1])
        else:
            train_dl = prep_data_loader(train_x, train_y, batch_size=cfg.TRAIN_BATCH_SIZE)
            valid_dl = prep_data_loader(val_x, val_y, batch_size=cfg.VALID_BATCH_SIZE)
        print('Data prepared!')
        return train_dl, valid_dl

    def prep_test_data(self, data, idenitity_phrases=False):
        print('Preparing data for testing...')
        if self.params['dataset_identifier'] == 'unintended_bias' and not idenitity_phrases:
            target_columns = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
            data = convert_labels(data, target_columns)
        test_data = data.copy()
        if self.params['use_bert']:
            self.initialize_tokenizer(data)
        if self.tokenizer is None:
            if not os.path.exists(self.tokenizer_filename):
                print('No tokenizer found. Fit a tokenizer first and then try again.')
            else:
                self.tokenizer = pickle.load(open(self.tokenizer_filename, 'rb'))
        if self.params['use_feature']:
            test_data = prep_features(test_data)
            self.features = list(set(test_data.columns) - set(data.columns))
            test_data[self.features] = test_data[self.features].fillna(0)
            test_data[self.features] = self.feature_normalizer.transform(test_data[self.features])
        test_data = clean_data(test_data)
        test_x, test_y = self.prepare_data(test_data)
        self.update_params({'n_test_samples': len(test_y)})
        if self.features is not None:
            test_stat = test_data[self.features].values
            test_dl = prep_data_loader(test_x, test_y,
                                       batch_size=cfg.VALID_BATCH_SIZE, stats=test_stat)
        elif self.params['use_bert']:
            test_dl = prep_data_loader(test_x[0], test_y, batch_size=cfg.VALID_BATCH_SIZE, masks=test_x[1])
        else:
            test_dl = prep_data_loader(test_x, test_y, batch_size=cfg.VALID_BATCH_SIZE)
        print('Data prepared!')
        return test_dl

    def prepare_data(self, data):
        if self.params['use_feature']:
            target_column = self.get_target()
            labels = data[target_column].tolist()
        else:
            if 'labels' not in data.columns:
                data['labels'] = data[list(data.columns)].sum(axis=1)
                data['labels'] = data['labels'].apply(lambda x: 1 if x > 1 else x)
                labels = data['labels'].tolist()
                data.drop(columns=['labels'])
            else:
                labels = data['labels'].tolist()
        return self.prep_text(data[cfg.TEXT_COLUMN].tolist()), labels

    def get_target(self):
        target_column = cfg.TOXICITY_COLUMN
        if self.params['dataset_identifier'] == 'unintended_bias':
            target_column = 'toxicity'
        elif self.params['dataset_identifier'] == 'wiki_talk_labels':
            target_column = 'labels'
        return target_column
