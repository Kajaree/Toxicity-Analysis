from IPython.display import display
import os
import pandas as pd
from sklearn.utils import resample
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.data.augment_data import DataAugment


def read_data(dataset_id, split):
    raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[dataset_id])
    data = pd.read_csv(os.path.join(raw_folder_path, f'{split}.csv'))
    data['length'] = data[cfg.TEXT_COLUMN].str.len()
    return data


if __name__ == '__main__':
    identities = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
                  'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                  'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion',
                  'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 'transgender',
                  'white']
    identity_terms_path = os.path.join('../../data', 'identity_terms.txt')
    with open(identity_terms_path) as f:
        DEBIAS_TERMS = [term.strip() for term in f.readlines()]
    dataset_id = 'unintended_bias'
    biased_data = read_data(dataset_id, 'train')
    target_columns = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
    for col in target_columns:
        biased_data[col] = biased_data[col].apply(lambda x: 1 if x else 0)
    biased_data = biased_data.fillna(0)
    biased_data['toxic'] = biased_data[target_columns].sum(axis=1)
    biased_data['toxic'] = biased_data['toxic'].apply(lambda x: True if x >= 1 else False)
    aug_dataset_id = 'wiki_talk_labels'
    aug_dataset = read_data(aug_dataset_id, 'train')
    aug_dataset = aug_dataset.rename(columns={"labels": 'toxic'})
    aug_dataset['toxic'] = aug_dataset['toxic'].values > 0.5
    # DEBIAS_TERMS = ['queer', 'gay', 'homosexual', 'lesbian', 'transgender']
    aug = DataAugment(terms=DEBIAS_TERMS)
    aug.add_deficit(aug_dataset)
    final_df_wiki = aug.postprocess_augmented_data(biased_data)
    final_df_wiki['toxic'] = final_df_wiki['toxic'].apply(lambda x: 1 if x else 0)
    debiased_data = pd.concat([aug_dataset, final_df_wiki])
    debiased_data = debiased_data.rename(columns={"toxic": 'labels'})
    raw_folder_path = os.path.join('../../data', cfg.DATASET_IDENTITY[aug_dataset_id])
    file_name = os.path.join(raw_folder_path, 'train_augmented.csv')
    debiased_data.to_csv(file_name, encoding='utf-8', index=False)
    # aug_dataset_id = 'toxicity'
    # aug_dataset = read_data(aug_dataset_id, 'train')
    # classes = list(aug_dataset.columns[2:])
    # aug_dataset['overall_toxic'] = aug_dataset[classes].sum(axis=1)
    # aug_dataset['overall_toxic'] = aug_dataset['overall_toxic'].apply(lambda x: True if x >= 1 else False)
    # final_df_tox = aug.postprocess_augmented_data(aug_dataset)
    # final_df_tox['toxic'] = final_df_tox['toxic'].apply(lambda x: 1 if x else 0)
    # target_columns = ['toxic', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
    # for col in target_columns:
    #     biased_data[col] = biased_data[col].apply(lambda x: 1 if x else 0)
    # biased_data = biased_data.fillna(0)
    # biased_data[identities] = (biased_data[identities].values > 0.5)
    # for identity in identities:
    #     biased_data[identity] = biased_data[identity].apply(lambda x: 1 if x else 0)
    # biased_data['over_all_toxic'] = biased_data[target_columns].sum(axis=1)
    # biased_data['identity_bias'] = biased_data[identities].sum(axis=1)
    # with_identity = biased_data[biased_data['identity_bias'] >= 1]
    # without_identity = biased_data[biased_data['identity_bias'] == 0]
    # with_identity_toxic = with_identity[with_identity['over_all_toxic'] >= 1]
    # print(len(with_identity_toxic))
    # with_identity_non_toxic = with_identity[with_identity['over_all_toxic'] == 0]
    # print(len(with_identity_non_toxic))
    # without_identity_non_toxic = without_identity[without_identity['over_all_toxic'] == 0]
    # print(len(without_identity_non_toxic))
    # without_identity_toxic = without_identity[without_identity['over_all_toxic'] >= 1]
    # print(len(without_identity_toxic))
    # columns = ['comment_text', 'over_all_toxic']
    # without_identity_toxic_sampled = resample(without_identity_toxic, replace=False,
    #                                           n_samples=25000, random_state=cfg.SEED)
    # without_identity_non_toxic_sampled = resample(without_identity_non_toxic, replace=False,
    #                                               n_samples=25000, random_state=cfg.SEED)
    # without_identity_sampled = pd.concat(without_identity_non_toxic_sampled, without_identity_toxic_sampled)
    # debiased_data = pd.concat(with_identity[columns], without_identity_sampled[columns])
    # debiased_data = debiased_data.rename(columns={"over_all_toxic": 'toxic'})
    # debiased_data = pd.concat(debiased_data, final_df_wiki[['comment_text', 'toxic']])
    # debiased_data = pd.concat(debiased_data, final_df_tox[['comment_text', 'toxic']])
    # debiased_data['toxic'] = debiased_data['toxic'].apply(lambda x: 1 if x > 1 else x)
    # print(debiased_data.shape[0])
    # file_name = os.path.join('../../data', 'augmented_data.csv')
    # debiased_data.to_csv(file_name, encoding='utf-8')