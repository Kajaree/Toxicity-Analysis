from IPython.display import display
import pandas as pd
import re
import copy
import unintended_bias_mitigation.utils.config as cfg


def parse_interval(interval):
    return int(round(float(interval.left))), int(round(float(interval.right)))


class TermDeficits:
    def __init__(self, term):
        self.term = term
        self.length_deficits = []

    def add_length_deficit(self, len_low, len_high, deficit):
        self.length_deficits.append([len_low, len_high, deficit])

    def total_deficit(self):
        return sum(x[2] for x in self.length_deficits)

    def has_deficit(self, length):
        for deficit in self.length_deficits:
            if deficit[0] <= length < deficit[1]:
                return deficit[2] > 0
        return False

    def fill_deficit(self, length):
        for deficit in self.length_deficits:
            if deficit[0] <= length < deficit[1]:
                deficit[2] -= 1
                break


class Deficits:
    def __init__(self, terms):
        """Construct a new Deficits manager for the given terms."""
        self.terms = terms
        # Used to determine if an example matches any term at all.
        self._terms_regex = re.compile('|'.join(terms), re.IGNORECASE)
        self.term_deficits = {term: TermDeficits(term) for term in terms}

    def add_deficit(self, term, length_low, length_high, deficit):
        self.term_deficits[term].add_length_deficit(length_low, length_high, deficit)

    def _term_matches(self, text):
        """Returns terms that are mentioned in the text (substring matching)."""
        # This is just for efficiency.
        if self._terms_regex.search(text) is None:
            return []
        return [term for term in self.terms if term in text.lower()]

    def accept_example(self, text):
        """If text can fill any remaining deficits, returns True and decrements all deficits
        that the text fills. Else, returns false."""
        # Does the text match any term?
        matched_terms = self._term_matches(text)
        if not matched_terms:
            return []
        # Do any of those terms have a deficit at this length?
        length = len(text)
        matched_term_deficits = [self.term_deficits[term] for term in matched_terms]
        if not any(term_deficit.has_deficit(length) for term_deficit in matched_term_deficits):
            return []

        # Decrement all matching term deficits.
        for term_def in matched_term_deficits:
            term_def.fill_deficit(length)
        return matched_terms

    def __deepcopy__(self, memo):
        copy_ = Deficits(self.terms)
        copy_.term_deficits = copy.deepcopy(self.term_deficits, memo)
        return copy_


def calculate_tox_ratio(df, isToxic=True):
    df = df.groupby(['intervals', 'toxic']).size().unstack('toxic').fillna(0)
    df['tox_ratio'] = df[isToxic] / (df[True] + df[False])
    return df


def compute_deficit(target_nontox_ratio, current_nontoxic, current_toxic):
    current_total = current_nontoxic + current_toxic
    return (target_nontox_ratio * current_total - current_nontoxic) / (1 - target_nontox_ratio)


class DataAugment:
    def __init__(self, terms, quantiles=4, train_split=0.6, test_split=0.2):
        self.terms = terms
        self.quantiles = quantiles
        self.train_split = train_split
        self.test_split = test_split
        self.deficits = Deficits(self.terms)
        self.len_intervals = None
        self.bins = None
        self.examples_to_keep = []
        self.matched_identities = []

    def add_deficit(self, frame):
        frame = frame.copy()
        self.len_intervals, self.bins = pd.qcut(frame['length'], q=self.quantiles, retbins=True)
        frame['intervals'] = self.len_intervals
        self.bins = self.bins.astype(int)
        print('bins:', self.bins)
        overall = frame.copy()
        overall = calculate_tox_ratio(overall)
        print('overall tox ratios:')
        display(overall)
        deficit_multiplier = 1 / self.train_split
        for term in self.terms:
            print(term)
            term_frame = frame[frame['comment_text'].str.lower().str.contains(term)]
            term_frame = calculate_tox_ratio(term_frame)
            for lenbin, row in term_frame.iterrows():
                low, high = parse_interval(lenbin)
                deficit = compute_deficit(
                    target_nontox_ratio=1 - overall.loc[lenbin, 'tox_ratio'],
                    current_nontoxic=row[False],
                    current_toxic=row[True])
                self.deficits.add_deficit(term, low, high, deficit * deficit_multiplier)
                term_frame.loc[lenbin, 'deficit'] = deficit

    def fill_deficits(self, examples):
        deficits = copy.deepcopy(self.deficits)
        examples_to_keep = []
        matched_identities = []
        for example in examples:
            matched_terms = deficits.accept_example(example)
            if len(matched_terms) > 0:
                examples_to_keep.append(example)
                matched_identities.append(matched_terms)
        print('using {} of {} ({:.1f})%)'.format(len(examples_to_keep), len(examples),
                                                 100 * len(examples_to_keep) / len(examples)))
        return examples_to_keep, matched_identities

    def postprocess_augmented_data(self, augment_data, target_label='toxic'):
        nontox_aug_examples = augment_data.query(f'{target_label} == False')
        nontox_df = self.augment_data(nontox_aug_examples['comment_text'].tolist(), isToxic=False)
        return nontox_df

    def augment_data(self, examples, isToxic=True):
        examples_to_keep, matched_identities = self.fill_deficits(examples)
        df = pd.DataFrame(examples_to_keep, columns=[cfg.TEXT_COLUMN])
        df[cfg.TOXICITY_COLUMN] = isToxic
        df[self.terms] = False
        for i, identity in enumerate(matched_identities):
            df.loc[i, identity] = True
        return df
