# Training file for MLP model
from datetime import datetime
import configargparse
from os.path import join
import os
import sys

import numpy as np
import torch

from src.utils.utils import str2bool, send_to_cuda, load_file_stores
from src.train.dataset import Dataset
from src.train.validator import Validator
from src.models.mlpmodel import MLPModel
from src.utils.logger import get_logger
from src.train.trainer import Trainer
from src.utils.file import FileObjectStore

np.warnings.filterwarnings('ignore')


def parse_args():
    parser = configargparse.ArgumentParser(description='Training Wikinet 2',
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # General
    general = parser.add_argument_group('General Settings.')
    general.add_argument('--my-config', required=True, is_config_file=True, help='config file path')
    general.add_argument('--exp_name', type=str, default="debug", help="Experiment name")
    general.add_argument("--debug", type=str2bool, default=True, help="whether to debug")

    # Data
    data = parser.add_argument_group('Data Settings.')
    data.add_argument('--data_path', required=True, type=str, help='location of data dir')
    data.add_argument('--data_type', type=str, help='name of train dataset, a directory of this name should contain '
                                                    'generated training data using gen_train_data.py')
    data.add_argument('--train_size', type=int, help='number of training abstracts')
    data.add_argument('--data_types', type=str, help='name of datasets separated by comma')

    # Max Padding
    padding = parser.add_argument_group('Max Padding for batch.')
    padding.add_argument('--max_context_size', type=int, help='max number of context')
    padding.add_argument('--max_ent_size', type=int, help='max number of entities considered in abstract')

    # Model Type
    model_selection = parser.add_argument_group('Type of model to train.')
    model_selection.add_argument('--pre_train', type=str, help='if specified, model will load state dict, must be ckpt')

    # Model params
    model_params = parser.add_argument_group("Parameters for chosen model.")
    model_params.add_argument('--dp', type=float, help='drop out')
    model_params.add_argument('--hidden_size', type=int, help='size of hidden layer in yamada model')

    # Candidate Generation
    candidate = parser.add_argument_group('Candidate generation.')
    candidate.add_argument("--num_candidates", type=int, default=32, help="Total number of candidates")
    candidate.add_argument("--prop_gen_candidates", type=float, default=0.5, help="Proportion of candidates generated")

    # Training
    training = parser.add_argument_group("Training parameters.")
    training.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    training.add_argument("--save_every", type=int, default=5, help="how often to checkpoint")
    training.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    training.add_argument("--batch_size", type=int, default=32, help="Batch size")
    training.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")
    training.add_argument('--lr', type=float, help='learning rate')
    training.add_argument('--wd', type=float, help='weight decay')
    training.add_argument('--embs_optim', type=str, choices=['adagrad', 'adam', 'rmsprop', 'sparseadam'],
                              help='optimizer for embeddings')
    training.add_argument('--other_optim', type=str, choices=['adagrad', 'adam', 'rmsprop'],
                              help='optimizer for paramaters that are not embeddings')
    training.add_argument('--sparse', type=str2bool, help='sparse gradients')

    # cuda
    parser.add_argument("--device", type=str, help="cuda device")
    parser.add_argument("--use_cuda", type=str2bool, help="use gpu or not")
    parser.add_argument("--profile", type=str2bool, help="if set will run profiler on dataloader and exit")

    args = parser.parse_args()
    logger = get_logger(args)

    if args.wd > 0:
        assert not args.sparse

    if args.use_cuda:
        devices = args.device.split(",")
        if len(devices) > 1:
            devices = tuple([int(device) for device in devices])
        else:
            devices = int(devices[0])
        args.__dict__['device'] = devices

    logger.info("Experiment Parameters:")
    print()
    for arg in sorted(vars(args)):
        logger.info('{:<15}\t{}'.format(arg, getattr(args, arg)))

    model_date_dir = join(args.data_path, 'models', '{}'.format(datetime.now().strftime("%Y_%m_%d")))
    if not os.path.exists(model_date_dir):
        os.makedirs(model_date_dir)
    model_dir = join(model_date_dir, args.exp_name)
    args.__dict__['model_dir'] = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return args, logger, model_dir


def setup(args, logger):

    print()
    logger.info("Loading word and entity embeddings from models/conll_v0.1.pt.....")
    state_dict = torch.load(join(args.data_path, 'models/conll_v0.1.pt'), map_location='cpu')['state_dict']
    ent_embs = state_dict['ent_embs.weight']
    word_embs = state_dict['word_embs.weight']
    logger.info("Model loaded.")

    logger.info("Loading filestore dicts.....")
    file_stores = load_file_stores(args.data_path)

    logger.info("Using {} for training.....".format(args.data_type))
    splits = ['train', 'dev', 'test']
    id2context = FileObjectStore(join(args.data_path, f'training_files/{args.data_type}/id2context'))
    split_examples = {split: [] for split in splits}
    for split in splits:
        split_examples[split] = FileObjectStore(join(args.data_path, f'training_files/{args.data_type}/{split}'))

    logger.info("Data loaded.")

    logger.info("Creating data loaders and validators.....")
    train_dataset = Dataset(id2context=id2context,
                            examples=split_examples['train'],
                            data_type=args.data_type,
                            args=args,
                            file_stores=file_stores)

    datasets = {}
    for data_type in args.data_types.split(','):
        datasets[data_type] = Dataset(id2context=id2context,
                                      examples=split_examples['dev'],
                                      data_type=args.data_type,
                                      args=args,
                                      file_stores=file_stores)
        logger.info(f"{data_type} dev dataset created.")

    return train_dataset, datasets, word_embs, ent_embs, file_stores


def get_model(args, word_embs, ent_embs, logger):

    model = MLPModel(word_embs=word_embs,
                     ent_embs=ent_embs,
                     args=args)

    if args.use_cuda:
        model = send_to_cuda(args.device, model)
    logger.info('Model created.')

    return model


def train(model=None,
          logger=None,
          datasets=None,
          train_dataset=None,
          args=None,
          file_stores=None,
          run=None):

    train_loader = train_dataset.get_loader(batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            drop_last=False)
    logger.info("Data loaders and validators created.There will be {} batches.".format(len(train_loader)))

    logger.info("Starting validation for untrained model.")
    validators = {}
    for data_type in args.data_types.split(','):
        loader = datasets[data_type].get_loader(batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)
        logger.info(f'Len loader {data_type} : {len(loader)}')
        validators[data_type] = Validator(loader=loader,
                                          args=args,
                                          data_type=data_type,
                                          run=run,
                                          file_stores=file_stores)

    trainer = Trainer(loader=train_loader,
                      args=args,
                      validator=validators,
                      model=model,
                      model_type='yamada',
                      profile=args.profile)
    logger.info("Starting Training:")
    print()
    best_model, best_results = trainer.train()
    logger.info("Finished Training")

    logger.info("Validating with best model....")
    best_model.eval()
    for data_type in args.data_types.split(','):
        validators[data_type].validate(best_model)


if __name__ == '__main__':
    Args, Logger, Model_dir = parse_args()
    Train_dataset, Datasets, Word_embs, Ent_Embs, File_stores = setup(Args, Logger)

    Model = get_model(Args, Word_embs, Ent_Embs, Logger)
    if Args.pre_train:
        Logger.info(f"loading pre trained model at models/{Args.pre_train}")
        state_dict = torch.load(join(Args.data_path, 'models', Args.pre_train), map_location='cpu')['state_dict']
        Model.load_state_dict(state_dict)
    train(model=Model,
          train_dataset=Train_dataset,
          datasets=Datasets,
          logger=Logger,
          args=Args,
          file_stores=File_stores)
