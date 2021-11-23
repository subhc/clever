import csv
import inspect
import json
import logging
import os
import random
import shutil
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import click
import torch
import transformers
from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.datasets import NLISentencesDatasetOnline
from modules.evaluators import EmbeddingSimilarityEvaluatorCustom, CUBDatasetEvaluator
from modules.losses import CrossEntropyLossCustom

from modules.models import SentenceTransformerCustom
from modules.models import SiameseSentenceTransformer


def log_details(args, params):
    logging.info("---------------------------------------")
    if args['eval_corpus'] == 'aab':
        logging.info('CUB evaluation using aab corpus')
    elif args['eval_corpus'] == 'wiki':
        logging.info('CUB evaluation using wiki corpus')
    logging.info("---------------------------------------\n")
    logging.info("---------------------------------------")
    logging.info("Arguments received: ")
    logging.info("---------------------------------------")
    for key, value in args.items():
        logging.info(f"{key:25}:  {value}")
    logging.info("---------------------------------------\n")
    logging.info("---------------------------------------")
    logging.info("Hyperparameters:")
    logging.info("---------------------------------------")
    for key, value in params.__dict__.items():
        if key.startswith('__'):
            continue
        logging.info(f"{key:25}:  {value}")
    logging.info("---------------------------------------")

def dump_experiment_details(save_path, args, files_list):
    shutil.copytree('./modules', f'{save_path}/modules')
    for files in files_list:
        if files is not None:
            for file in files:
                os.makedirs(os.path.dirname(os.path.join(save_path, file)), exist_ok=True)
                shutil.copy2(file, os.path.join(save_path, file))
    shutil.copy2(os.path.abspath(inspect.stack()[0][1]), save_path)
    with open(save_path + '/args.json', 'w') as f:
        json.dump(args, f)

# sbert models are here: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/

@click.command()
@click.option('--experiment', default=None, help='Name of the experiment')
@click.option('--files', default='dummy', help='The triplet files to be processed (comma separated)')
@click.option('--pred-source', '--pred_source', default='sup_aoanet,sup_sat')
@click.option('--dataset', default='CUB', help='CUB/FLO')
@click.option("--nolog", is_flag=True)
@click.option("--sbert_model", "--sbert-model", type=str, default='roberta-large-nli-mean-tokens')  # bert-large-nli-mean-tokens
@click.option("--fastdebug", is_flag=True)
@click.option("--eval-corpus", "--eval_corpus", default='aab', help='aab/wiki')
@click.option('--task', default='nli', help='triplet/nli')
@click.option('--scheduler', default='constantlr', help='warmuplinear/constantlr/warmupconstant/warmuplinear/warmupcosine/warmupcosinewithhardrestarts')
@click.option("--warmup_epochs", "--warmup-epochs", "-w", type=float, default=0)
@click.option('--optimizer_class', '--optimizer-class', default='adam', help='adam/sgd')
@click.option("--batch-size", "--batch_size", "-b", type=int, default=16)
@click.option("--epochs", "-n", type=int, default=-1)
@click.option("--training_seed", type=int, default=1729)
@click.option("--encoder-lr", "--encoder_lr", type=float, default=5e-6)
@click.option("--fc-lr", "--fc_lr", type=float, default=1e-5)
@click.option("--encoder-weight-decay", "--encoder_weight_decay", type=float, default=.01)
@click.option("--fc-weight-decay", "--fc_weight_decay", type=float, default=0.0)
@click.option("--no_fc", "--no-fc", is_flag=True)
@click.option("--eval-method", "--eval_method", default='classification', help='adam/sgd')
@click.option("--train-split", "--train_split", default='supervised_train')
@click.option("--val-split", "--val_split", default='supervised_val')
@click.option("--eval-data", "--eval_data", default='captions_pred', help='attr_gt_class/attr_pred_image/attr_gt_image/captions_gt/captions_pred')
@click.option("--training-data", "--training_data", default='captions_gt', help='wiki/aab/captions_gt/captions_pred')
@click.option("--cache-name", "--cache_name", default='roberta', help='roberta.v1/roberta.v2')
@click.option("--evaluation_steps", type=int, default=5)
def main(experiment, files, pred_source, dataset, nolog, fastdebug, eval_corpus, sbert_model, scheduler, warmup_epochs, task, optimizer_class,
         batch_size, epochs, training_seed, encoder_lr, fc_lr, encoder_weight_decay, fc_weight_decay, no_fc,
         eval_method, train_split, val_split, eval_data, training_data, cache_name, evaluation_steps):
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)
    args = inspect.currentframe()
    args = inspect.getargvalues(args)
    args = dict([(k, v) for k, v in args.locals.items() if k in args.args])

    train_files = [f'./{dataset}/txts/{task}/train_{fn}.txt' for fn in files.split(',')] if files != 'dummy' else None
    val_files = [f'./{dataset}/txts/{task}/val_{fn}.txt' for fn in files.split(',')] if files != 'dummy' else None

    if nolog:
        experiment = ""
        output_path = None
        writer = None
        handlers = [logging.StreamHandler()]

    else:
        datetime_ = datetime.now()
        experiment = experiment + '/' if experiment is not None else "%Y%m%d_%H%M%S"
        args['experiment'] = experiment
        output_path = f'./output/{experiment}/{datetime_.strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(f'./runs/{experiment + "/" if experiment is not None else ""}/{datetime_.strftime("%Y%m%d_%H%M%S")}')
        save_path = os.path.join(output_path, 'logs')
        dump_experiment_details(save_path, args, [train_files, val_files])
        handlers = [logging.FileHandler(f'{save_path}/main.log', mode='w'), logging.StreamHandler()]


    logging.basicConfig(handlers=handlers, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.getLogger('transforms').setLevel(logging.ERROR)

    # Hyperparameters
    params = dict(train_batch_size=batch_size,
                  pred_source=pred_source,
                  eval_corpus=eval_corpus,
                  task=task,
                  num_epochs=epochs if epochs > 0 else (110 if dataset == 'CUB' else 120),
                  scheduler=scheduler,
                  optimizer_class=transformers.AdamW if optimizer_class == 'adam' else torch.optim.SGD,
                  optimizer_params={'lr': fc_lr} if optimizer_class == 'adam' else {'lr': fc_lr, 'momentum': 0.9},
                  warmup_epochs=warmup_epochs if warmup_epochs >= 0 else 1,
                  encoder_lr=encoder_lr,
                  fc_lr=fc_lr,
                  encoder_weight_decay=encoder_weight_decay,
                  fc_weight_decay=fc_weight_decay,
                  evaluation_steps=(5 if not fastdebug else 1) if evaluation_steps == -1 else evaluation_steps,
                  fc_hidden_dim=256,
                  fc_out_dim=64,
                  eval_method=eval_method,
                  ce_weight=None,
                  eval_data=eval_data,
                  train_split=train_split,
                  val_split=val_split,
                  cache_name=cache_name if cache_name is not None else f'roberta.{train_split}',
                  dataset=dataset,
                  training_data=training_data,
                  num_workers=5)

    params = SimpleNamespace(**params)
    if fastdebug:
        sbert_model = sbert_model.replace('large', 'base')
    sbert = SentenceTransformer(sbert_model)

    sbert_modules = [sbert[0].train(), sbert[1].train()]
    if not no_fc:
        fc_in_dim = sbert[1].pooling_output_dimension  # 1024 for large, 738 for base
        fc_module1 = models.Dense(in_features=fc_in_dim, out_features=params.fc_hidden_dim, bias=True, activation_function=nn.Tanh())
        fc_module2 = models.Dense(in_features=params.fc_hidden_dim, out_features=params.fc_out_dim, bias=True, activation_function=nn.Tanh())
        sbert_modules += [fc_module1.train(), fc_module2.train()]

    log_details(args, params)
    logging.info("Experiment name: " + experiment)

    root = '.'
    model = SentenceTransformerCustom(modules=sbert_modules)
    model = SiameseSentenceTransformer(sbert_model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

    logging.info(f"Training {dataset}  {params.task} {params.eval_method}")

    train_data = NLISentencesDatasetOnline(root=root, model_name=params.cache_name, dataset=params.dataset, split=params.train_split, fastdebug=fastdebug,
                                           model=model, train_data=params.training_data, eval_method=params.eval_method)

    train_loss = CrossEntropyLossCustom(model=model, ce_weight=params.ce_weight)
    shuffle = False

    train_dataloader = DataLoader(train_data, shuffle=shuffle, batch_size=params.train_batch_size, num_workers=0 if fastdebug else params.num_workers)

    if params.val_split != '':
        val_data = NLISentencesDatasetOnline(root=root, model_name=params.cache_name, dataset=params.dataset, split=params.val_split, fastdebug=fastdebug, model=model.get_sbert(),
                                             train_data=params.training_data, eval_method=params.eval_method)

        val_dataloader = DataLoader(val_data, shuffle=False, batch_size=params.train_batch_size)

    evaluators = [EmbeddingSimilarityEvaluatorCustom(train_dataloader, writer=writer, prefix=params.train_split, noun_vocab=train_data.get_noun_vocab())]
    if params.val_split != '':
        evaluators.append(EmbeddingSimilarityEvaluatorCustom(val_dataloader, writer=writer, noun_vocab=train_data.get_noun_vocab()))

    if 'supervised' in params.train_split:
        evaluators = []
        evaluators.append(CUBDatasetEvaluator(split='supervised_test', type='gzsl', writer=writer, args=args, all_scores=True))
    else:
        evaluators = []
        evaluators.append(CUBDatasetEvaluator(split='test_unseen', type='zsl', writer=writer, args=args, all_scores=True))
        evaluators.append(CUBDatasetEvaluator(split='test_seen', type='gzsl', writer=writer, args=args, all_scores=True))
        evaluators.append(CUBDatasetEvaluator(split='test_unseen', type='gzsl', writer=writer, args=args, all_scores=True))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluators,
              optimizer_params=params.optimizer_params,
              lr={'encoder': params.encoder_lr, 'fc': params.fc_lr},
              weight_decay={'encoder': params.encoder_weight_decay, 'fc': params.fc_weight_decay},
              epochs=params.num_epochs,
              scheduler=params.scheduler,
              optimizer_class=params.optimizer_class,
              evaluation_steps=params.evaluation_steps,
              warmup_epochs=params.warmup_epochs,
              output_path=output_path,
              writer=writer,
              task=params.task,
              args=args)

if __name__ == '__main__':
    main()
