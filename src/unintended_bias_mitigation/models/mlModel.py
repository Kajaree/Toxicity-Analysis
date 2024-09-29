import os
import re
import tqdm
import string
import unicodedata
from nltk.tokenize import word_tokenize
from IPython.display import display
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import unintended_bias_mitigation.utils.config as cfg
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from unintended_bias_mitigation.evaluation.metrics import Metrics
from unintended_bias_mitigation.evaluation.nuancedMetrics import NuancedMetric


import warnings
warnings.filterwarnings("ignore")

punctuations = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^','®', '`', '<', '→', '°', '€',
                '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►',
                '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒',
                '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
                '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣',
                '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}


def get_misspell():
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re


def strip_ip(s):
    ip = re.compile('(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}'
                    + '(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))')
    try:
        found = ip.search(s)
        return s.replace(found.group(), ' ')
    except:
        return s


def clean_punctuations(x):
    x = str(x)
    for punctuation in punctuations + list(string.punctuation):
        if punctuation in x:
            x = x.replace(punctuation, f' ')
    return x


def replace_misspell(text):
    misspellings, misspellings_re = get_misspell()

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)


def remove_non_ascii(word):
    return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def get_emoticon_token(word_token):
    a=[":-)", ":)", "^^", "^.^","🙏","👏","😊","💐","👍","🤗","😇","❤️",'😍',"🥰","☺️","😁","😂"]
    if any(x in word_token for x in a):
        return "emo_happy"
    elif word_token in [":-(", ":("]:
        return "emo_sad"
    elif word_token in [":-D", ":D"]:
        return "emo_laugh"
    elif word_token in ["^_^", ":-|", ":|"]:
        return "emo_neutr"
    elif word_token in [":-o", ":-O", ":-0", ":o", ":O", ":0"]:
        return "emo_surpr"
    elif word_token in [";)", ";-)"]:
        return "emo_wink"
    elif word_token in [":-P", ":P", ":-p", ":-p"]:
        return "emo_tongue"
    elif word_token == "<3":
        return "emo_heart"
    else:
        return word_token


def clean_text(text):
    replace_misspell(text)
    text = strip_ip(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', ' ')
    text = text.replace("\r", ' ')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = clean_punctuations(text)
    # Remove whitespaces
    text = text.lstrip().rstrip()
    tokens = word_tokenize(text)
    tokens_preprocessed = [remove_non_ascii(word) for word in tokens]
    word_tokens = [get_emoticon_token(word) for word in tokens_preprocessed]
    return ' '.join(word_tokens)


def clean_data(data):
    word_tokens = data['comment_text'].apply(lambda s: clean_text(s))
    data['tokenized_words'] = word_tokens.to_list()
    data['tokenized_words'] = data['tokenized_words'].fillna('_##_')
    return data


def read_data(dataset_identifier, train_file='train', test_file='test'):
    raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[dataset_identifier])
    train = pd.read_csv(os.path.join(raw_folder_path, f"{train_file}.csv"))
    targets = list(train.columns[2:])
    if dataset_identifier == 'unintended_bias':
        train = train.rename(columns={"target": 'toxicity'})
        targets = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        train = convert_labels(train, targets.copy())
        train = balanced_sample(train, targets)
        test = pd.read_csv(os.path.join(raw_folder_path, "test_public_expanded.csv"))
        test = convert_labels(test, targets.copy())
    else:
        test = pd.read_csv(os.path.join(raw_folder_path, f"{test_file}.csv"))
    if dataset_identifier != 'wiki_talk_labels':
        train = add_label(train, targets)
        test = add_label(test, targets)
    return train, test


def add_label(df, targets):
    if 'labels' not in df.columns:
        df['labels'] = df[targets].sum(axis=1)
        df['labels'] = df['labels'].apply(lambda x: 1 if x > 1 else x)
    return df


def save_test_predictions(dataset_identifier, predictions, scores, model_name, idenitity_phrases=False):
    raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[dataset_identifier])
    filename = 'test_predictions'
    if idenitity_phrases:
        filename = 'identity_predictions'
    prediction_file = os.path.join(raw_folder_path, f"{filename}.csv")
    if not os.path.isfile(prediction_file):
        test_df = pd.read_csv(os.path.join(raw_folder_path, "test.csv"))
    else:
        test_df = pd.read_csv(prediction_file)
    test_df[f"{model_name}_Predictions"] = predictions.tolist()
    test_df[f"{model_name}_Scores_0"] = scores[:, 0].tolist()
    test_df[f"{model_name}_Scores_1"] = scores[:, 1].tolist()
    test_df.to_csv(prediction_file, index=False)
    return test_df


def convert_labels(data, target_columns):
    data[target_columns] = (data[target_columns].values > 0.5).astype(int)
    target_columns.append(cfg.TEXT_COLUMN)
    return data[target_columns]


def balanced_sample(dataset_df, targets):
    if 'labels' not in dataset_df.columns:
        dataset_df['labels'] = dataset_df[targets].sum(axis=1)
        dataset_df['labels'] = dataset_df['labels'].apply(lambda x: 1 if x > 1 else x)
    df_toxic = dataset_df[dataset_df['labels'] == 1]
    df_non_toxic = dataset_df[dataset_df['labels'] == 0]
    df_toxic_sampled = resample(df_toxic, replace=True, n_samples=df_toxic.shape[0], random_state=cfg.SEED)
    df_non_toxic_sampled = resample(df_non_toxic, replace=False, n_samples=cfg.SAMPLE_SIZE, random_state=cfg.SEED)
    df = pd.concat([df_non_toxic_sampled, df_toxic_sampled])
    return df.sample(frac=1, random_state=cfg.SEED)


def lr(training_sequence, training_target):
    vectorizer_params = dict(min_df=5, max_df=0.8)
    pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('chi', SelectKBest(chi2, k=1200)),
        ('clf', LogisticRegression(random_state=cfg.SEED)),
    ])
    model = pipeline.fit(training_sequence, training_target)
    return model


def mnb(training_sequence, training_target):
    vectorizer_params = dict(min_df=5, max_df=0.8)
    pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    model = pipeline.fit(training_sequence, training_target)
    return model


def build_model(dataset_identifier, model_name, use_augmented=False, idenitity_phrases=False):
    train_file = 'train'
    test_file = 'test'
    if use_augmented:
        train_file = f"{train_file}_augmented"
        model_name = f"{model_name}_augmented"
    if idenitity_phrases:
        test_file = "identity_phrase_templates"
    train, test = read_data(dataset_identifier, train_file=train_file, test_file=test_file)
    train = clean_data(train)
    test = clean_data(test)
    training_sequence = train['tokenized_words']
    training_target = train['labels']
    test_target = test['labels']
    test_seq = test['tokenized_words']
    if model_name == 'lr':
        model = lr(training_sequence, training_target)
    else:
        model = mnb(training_sequence, training_target)
    predicted_target = model.predict(test_seq)
    scores = model.predict_proba(test_seq)
    test_df = save_test_predictions(dataset_identifier, predicted_target, scores,
                                    model_name, idenitity_phrases=idenitity_phrases)
    evaluate_model(test_df, dataset_identifier, model_name, predicted_target,
                   test_target, scores, idenitity_phrases=idenitity_phrases)


def evaluate_model(test_df, dataset_identifier, model_name, predicted_target,
                   test_target, scores, idenitity_phrases=False):
    identity_terms_path = os.path.join('../../data', 'identity_terms.txt')
    with open(identity_terms_path) as f:
        DEBIAS_TERMS = [term.strip() for term in f.readlines()]
    bias_eval_file = f"bias_analysis_{dataset_identifier}"
    eval_filename = f"evaluation_{dataset_identifier}"
    if idenitity_phrases:
        bias_eval_file = f"bias_analysis_{dataset_identifier}_identity"
        eval_filename = f"evaluation_{dataset_identifier}_identity"
    test_metric = Metrics(test_target, predicted_target, scores[:, 1])
    test_metric.save_evaluation_report(model_name=model_name,
                                       filename=eval_filename)
    bias_metrics = NuancedMetric(data=test_df, subgroups=DEBIAS_TERMS,
                                 model_names=[model_name])
    bias_metrics.evaluate_model(bias_metrics_file=bias_eval_file, eval_filename=eval_filename)


