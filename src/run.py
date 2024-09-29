import os
import argparse
import unintended_bias_mitigation.utils.config as cfg
from unintended_bias_mitigation.models.baseModel import BaseModel
from unintended_bias_mitigation.models.twdModel import TWDModel
from unintended_bias_mitigation.models.mlModel import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for toxicity detection')
    parser.add_argument('--model_class', choices=['LSTM', 'CNN', 'BILSTM', 'BERT'], default='LSTM')
    parser.add_argument('--dataset_identifier', choices=['toxicity', 'unintended_bias',
                                                         'wiki_talk_labels', 'generated_data'],
                        default='toxicity')
    parser.add_argument('--identity_phrase_testing', type=bool, default=False)
    parser.add_argument('--use_embedding', type=bool, default=False)
    parser.add_argument('--use_augmented', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--twd_testing', type=bool, default=False)
    parser.add_argument('--ML', type=bool, default=False)
    parser.add_argument('--enhance_model_name', choices=['lr', 'mnb'], default=None)
    parser.add_argument("--model_ensemble", nargs="+", default=["LSTM_1", "LSTM_1_featured"])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    params = {'model_class': args.model_class, 'model_name': None}
    if args.train and not args.ML:
        learning_rates = [0.001, 0.002, 0.0001, 0.0002, 0.0005]
        if args.model_class == 'BERT':
            learning_rates = [5e-5, 3e-5, 2e-5]
        training_params = getattr(cfg, "DEFAULT_TRAIN_PARAMS").copy()
        dataset_params = getattr(cfg, "DEFAULT_DATSET_PARAMS").copy()
        model_params = getattr(cfg, f"DEFAULT_{args.model_class}_HPARAMS").copy()
        dataset_params['dataset_identifier'] = args.dataset_identifier
        if args.model_class == 'BERT':
            dataset_params['use_bert'] = True
        if dataset_params['dataset_identifier'] == 'unintended_bias':
            dataset_params['test_file'] = 'test_public_expanded'
        if dataset_params['dataset_identifier'] == 'generated_data':
            dataset_params['train_file'] = 'augmented_data'
        if args.use_augmented:
            dataset_params['dataset_identifier'] = 'wiki_talk_labels'
            dataset_params['train_file'] = 'train_augmented'
        model_params.update(params)
        for i in range(len(learning_rates)):
            training_params['learning_rate'] = learning_rates[i]
            model_name = f"{args.model_class}_{(i + 1)}"
            if args.use_augmented:
                model_name = f"{model_name}_augmented"
            if args.use_embedding:
                training_params['use_embedding'] = True
                model_name = f"{model_name}_embedded"
                model_params['embedding_dim'] = 100
            print(f"Model name: {model_name}")
            base_model = BaseModel(model_params=model_params.copy(),
                                   training_params=training_params.copy(),
                                   dataset_params=dataset_params.copy())
            base_model.build_model(f"{model_name}")
            if dataset_params['dataset_identifier'] != 'generated_data':
                base_model.evaluate_model()

    if args.train and args.ML:
        model_names = ['lr', 'mnb']
        for model_name in model_names:
            build_model(args.dataset_identifier, model_name,
                        use_augmented=args.use_augmented,
                        idenitity_phrases=args.identity_phrase_testing)

    if args.test:
        model_dir = os.path.join(cfg.DEFAULT_MODEL_DIR, f"{cfg.DATASET_IDENTITY[args.dataset_identifier]}")
        model_files = os.listdir(model_dir)
        dataset_params = {'dataset_identifier': args.dataset_identifier}
        for model_file in model_files:
            model_name = model_file.replace('.pt', '')
            if args.model_class in model_name:
                print(model_name)
                params['model_name'] = model_name
                base_model = BaseModel(model_params=params.copy(),
                                       dataset_params=dataset_params.copy())
                base_model.evaluate_model(idenitity_phrases=args.identity_phrase_testing)

    if args.twd_testing and not args.ML:
        dataset_params = {'dataset_identifier': args.dataset_identifier}
        enhanced_params = params.copy()
        params['model_name'] = args.model_ensemble[0]
        enhanced_params['model_name'] = args.model_ensemble[1]
        base_class = params['model_name'].split('_')[0]
        enhanced_class = enhanced_params['model_name'].split('_')[0]
        base_model_params = getattr(cfg, f"DEFAULT_{base_class}_HPARAMS").copy()
        enhanced_model_params = getattr(cfg, f"DEFAULT_{enhanced_class}_HPARAMS").copy()
        params['model_class'] = base_class
        enhanced_params['model_class'] = enhanced_class
        base_model_params.update(params)
        enhanced_model_params.update(enhanced_params)
        base_model = BaseModel(model_params=base_model_params.copy(),
                               dataset_params=dataset_params.copy())
        enhanced_model = BaseModel(model_params=enhanced_model_params.copy(),
                                   dataset_params=dataset_params.copy())
        thresholds = cfg.BOUNDARY_THRESHOLDS
        for th in thresholds:
            twd_model = TWDModel(base_model, enhanced_model, boundary_threshold=th)
            twd_model.evaluate_model(idenitity_phrases=args.identity_phrase_testing)

    if args.twd_testing and args.ML:
        dataset_params = {'dataset_identifier': args.dataset_identifier}
        params['model_name'] = args.model_ensemble[0]
        base_class = params['model_name'].split('_')[0]
        base_model_params = getattr(cfg, f"DEFAULT_{base_class}_HPARAMS").copy()
        params['model_class'] = base_class
        base_model_params.update(params)
        base_model = BaseModel(model_params=base_model_params.copy(),
                               dataset_params=dataset_params.copy())
        thresholds = cfg.BOUNDARY_THRESHOLDS
        for th in thresholds:
            twd_model = TWDModel(base_model, boundary_threshold=th)
            twd_model.evaluate_model(enhance_model_name=args.model_ensemble[1],
                                     idenitity_phrases=args.identity_phrase_testing)






