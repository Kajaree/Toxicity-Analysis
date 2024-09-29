MODEL_CHECKPOINT_FOLDER = "../../pretrained_models/"
TEMPORARY_CHECKPOINTS_PATH = 'temporary_checkpoints/'
MAX_SENTENCE_LENGTH = 250
MAX_NUM_WORDS = 10000
TOP_WORDS = 5000
SEQUENCE_LENGTH = 4
SEED = 666
TRAIN_PERCENT = 0.8
VALIDATE_PERCENT = 0.2
BATCH_SIZE = 256
MAX_EPOCHS = 3
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
TRAIN_BATCH_SIZE = 20
VALID_BATCH_SIZE = 10
TEXT_COLUMN = 'comment_text'
TOXICITY_COLUMN = 'toxic'
BIAS_DATA_FOLDER = '../../data/bias-in-toxicity/'
TOXICITY_DATA_FOLDER = '../../data/jigsaw-toxic-comment/'
SAMPLE_INDICES = '../../data/sampling_indices/'
DEFAULT_EMBEDDINGS_PATH = '../../data/embeddings/glove.6B.100d.txt'
DEFAULT_MODEL_DIR = '../models'
DEFAULT_PLOTS_DIR = '../plots'
DEFAULT_TOKENIZER_DIR = '../tokenizers'
DEFAULT_HPARAMS_DIR = '../hyperParams'
DEFAULT_EVALS_DIR = '../evaluation_reports'
DEFAULT_BOW_DIR = '../vectorized_data/BoW'
DEFAULT_TFIDF_DIR = '../vectorized_data/TfIdf'
SAMPLE_SIZE = 200000
TOX_RATIO = 0.2
DEFAULT_NORMALIZER_DIR = '../normalizers'
DEFAULT_EVAL_METRICS = ['accuracy', 'f1_score', 'aupr_score', 'auroc_score', 'overall_bias_score']
IDENTITY_TERMS = ['queer', 'gay', 'homosexual', 'lesbian', 'transgender', 'asian', 'atheist', 'bisexual', 'black',
                  'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'jewish', 'latino', 'male', 'muslim']

BOUNDARY_THRESHOLDS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                       0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]
DEFAULT_LSTM_HPARAMS = {
    'dropout_rate': 0.3,
    'output_size': 2,
    'embedding_dim': 400,
    'hidden_dim': 256,
    'n_layers': 2,
}
DATASET_IDENTITY = {
    'toxicity': 'jigsaw-toxic-comment/',
    'unintended_bias': 'bias-in-toxicity/',
    'wiki_talk_labels': 'wikipedia_talk_labels/',
    'generated_data': 'generated_data/'
}
DEFAULT_CNN_HPARAMS = {
    'embedding_dim': 400,
    'output_size': 2,
    'hidden_dim': 256,
    'dropout_rate': 0.3,
    'cnn_filter_size': 128,
    'cnn_kernel_size': 5,
    'cnn_pooling_size': 5
}
DEFAULT_DATSET_PARAMS = {
    'dataset_identifier': 'toxicity',
    'max_sequence_length': 250,
    'max_num_words': 10000,
    'split_ratio': 0.8,
    'train_file': 'train',
    'test_file': 'test',
    'tox_ratio': 0.2,
    'use_feature': False,
    'use_bert': False
}
DEFAULT_TRAIN_PARAMS = {
    'clip': 5,
    'epochs': 50,
    'learning_rate': 0.0002,
    'criterion': 'CrossEntropyLoss',
    'optimizer': 'RMSprop',
    'use_class_weight': False,
    'use_embedding': False
}
DEFAULT_VECTORIZER_IDENTITY = {
    'bow': 'CountVectorizer',
    'tfidf': 'TfidfVectorizer'
}

BOUNDARY_LABELS = {
    'toxic': -1,
    'non_toxic': -2
}

DEFAULT_BILSTM_HPARAMS = {
    'dropout_rate': 0.3,
    'output_size': 2,
    'embedding_dim': 400,
    'hidden_dim': 256,
    'n_layers': 2,
}
DEFAULT_BERT_HPARAMS = {
    "model_checkpoint": "distilbert-base-uncased",
    "batch_size": 16,
    'output_size': 2,
    'dropout_rate': 0.3,
    'hidden_dim': 256,
}
