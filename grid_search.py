# Main training file
import pickle
from os.path import join
import gc
import time

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import pandas as pd

import torch

from train import parse_args, setup, get_model
from src.utils.utils import load_file_stores
from src.train.trainer import Trainer
from src.train.validator import Validator

np.warnings.filterwarnings('ignore')


def grid_search(word_embs=None,
                ent_embs=None,
                logger=None,
                datasets=None,
                model_dir=None,
                train_dataset=None,
                args=None):
    param_grid = {'dp': [0.1, 0.2, 0.3, 0.5, 0.8],
                  'hidden_size': [1000, 2000, 3000],
                  'lr': [1e-2, 5e-2, 1e-3, 5e-2],
                  'wd': [1e-4, 1e-5],
                  'num_candidates': [128, 256],
                  'prop_gen_candidates': [0.5, 0.75, 1],
                  'other_optim': ['rmsprop', 'adam']
                  }
    grid_results_dict = {}
    pd_results = list()
    data_types = args.data_types.split(',')

    for param_dict in list(ParameterSampler(param_grid, 50)):
        for k, v in param_dict.items():
            assert k in args.__dict__
            args.__dict__[k] = v

        for dataset in list(datasets.values()) + [train_dataset]:
            dataset.num_cand_gen = int(param_dict['num_candidates'] * param_dict['prop_gen_candidates'])
            dataset.num_candidates = param_dict['num_candidates']

        model = get_model(args, word_embs, ent_embs, logger)
        train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)
        logger.info("Train loader created.There will be {} batches.".format(len(train_loader)))

        logger.info("GRID SEARCH PARAMS : {}".format(param_dict))
        result_key = tuple(param_dict.items())
        grid_results_dict[result_key] = {data_type: [] for data_type in data_types}

        logger.info("Starting validation for untrained model.....")
        validators = {}
        for data_type in data_types:
            loader = datasets[data_type].get_loader(batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    drop_last=False)
            logger.info(f'Len loader {data_type} : {len(loader)}')
            validators[data_type] = Validator(loader=loader,
                                              args=args,
                                              dicts=File_stores)

        trainer = Trainer(loader=train_loader,
                          args=args,
                          validator=validators,
                          model=model,
                          model_type='yamada',
                          grid_results_dict=grid_results_dict,
                          result_key=result_key)

        logger.info("Starting Training.....")
        print()
        best_model, best_results = trainer.train()
        logger.info("Finished Training")

        pd_results.append({**param_dict, **best_results})
        print('PD RESULTS: {}'.format(pd_results))
        df = pd.DataFrame(pd_results)
        df.to_csv(join(model_dir, 'hyper_df.csv'))

        for k, v in grid_results_dict.items():
            print(k)
            print(v)

        with open(join(model_dir, 'grid_search_results.pickle'), 'wb') as f:
            pickle.dump(grid_results_dict, f)

        del model, trainer, train_loader, loader, validators, best_model, best_results
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)

    return grid_results_dict, pd_results


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_dataset, Datasets, Word_embs, Ent_embs, File_stores = setup(Args, Logger)
    result_dict, pd_dict = grid_search(word_embs=Word_embs,
                                       ent_embs=Ent_embs,
                                       model_dir=Model_dir,
                                       train_dataset=Train_dataset,
                                       datasets=Datasets,
                                       logger=Logger,
                                       args=Args)
    df = pd.DataFrame(pd_dict)
    df.to_csv(join(Model_dir, 'hyper_df.csv'))
