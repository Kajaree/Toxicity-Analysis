import re
import string
import unintended_bias_mitigation.utils.config as cfg


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
    return text.lstrip().rstrip()


def clean_data(data):
    data[cfg.TEXT_COLUMN] = data[cfg.TEXT_COLUMN].apply(clean_text)
    data[cfg.TEXT_COLUMN] = data[cfg.TEXT_COLUMN].fillna('_##_')
    return data


