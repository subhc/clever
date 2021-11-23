import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import glob
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from custom_dataset import ImageTensorDataset, ThreeClassTensorDataset
from new_loss import CrossEntropyLoss
from siamese_fcn import SiameseFCN
from utils import accuracy, rank

import wandb
def log_details(args):
    logging.info("---------------------------------------")
    logging.info("Arguments received: ")
    logging.info("---------------------------------------")
    for k, v in sorted(args.__dict__.items()):
        logging.info(f"{k:25}: {v}")
    logging.info("---------------------------------------\n")


def initialize_logging(args):
    if args.fastdebug:
        writer = None
        output_path = None
        handlers = [logging.StreamHandler()]
    else:
        datetime_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = args.experiment
        experiment += '/' + "_".join([f"{key}={value}" for key, value in sorted(args.__dict__.items()) if key not in['experiment', 'gpu']])
        output_path = f'./output/{experiment}/{datetime_}'
        save_path = os.path.join(output_path, 'logs')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for file in glob.glob(r'*.py'):
            shutil.copy(file, save_path)
        with open(save_path + '/args.json', 'w') as f:
            json.dump(args.__dict__, f)
        writer = SummaryWriter(f'./runs/{experiment + "/" if experiment is not None else ""}/{datetime_}')
        handlers = [logging.FileHandler(f'{save_path}/main.log', mode='w'), logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    return writer, output_path


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on image features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', metavar='BATCH_SIZE', type=int, default=32,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-c', '--corpus', metavar='CORPUS', type=str, default=None,
                        help='Corpus aab/wiki', dest='corpus')
    parser.add_argument('-d', '--dataset', metavar='DATASET', type=str, default='CUB',
                        help='Dataset CUB/FLO', dest='dataset')
    parser.add_argument('-e', '--epochs', metavar='EPOCHS', type=int, default=15000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-f', '--feature-model', metavar='MODEL', type=str, default='FGSM',
                        help='Model used to extract feature roberta/ours', dest='feature_model')
    parser.add_argument('-g', '--gpu', metavar='GPU', type=int, nargs='?', default=0,
                        help='GPU ID', dest='gpu')
    parser.add_argument('-k', metavar='K', type=int, default=0,
                        help='Nearest neighbours, k=0 to switch off', dest='k')
    parser.add_argument('-l', '--learning_rate', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--training-seed', metavar='SEED', type=int, default=1234,
                        help='Training seed', dest='training_seed')
    parser.add_argument('-t', '--loss_type', metavar='TYPE', type=str, default='normal',
                        help='Loss type: normal, doublesoftmax, max', dest='loss_type')
    parser.add_argument('-w', '--num_workers', dest='num_workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('-x', '--experiment', dest='experiment', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('-y', '--weight-decay', metavar='WD', type=float, default=0.0,
                        help='Weight decay', dest='weight_decay')
    parser.add_argument('--fastdebug', dest='fastdebug', action='store_true')
    parser.add_argument('--evaluation_steps', dest='evaluation_steps', type=int, default=50,
                        help='Number of workers')
    parser.add_argument('--three', dest='three', type=bool, default=False,
                        help='3 class')
    parser.add_argument('--loss_mult', metavar='LM', type=float, default=8,
                        help='Loss multiplier', dest='loss_mult')
    parser.add_argument('-p', metavar='RATIO', type=float, default=0.5,
                        help='Negative-Neutral ratio', dest='p')
    parser.add_argument('--load-checkpoint', metavar='CHECKPOINT', type=str, default=None,
                        help='pth model path', dest='load_checkpoint')
    return parser.parse_args()


def main(args=None):
    torch.cuda.set_device(f"cuda:{args.gpu}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.gpu)
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)


    cfg = args.__dict__
    wandb.init(project='FGSM', entity='my_user')
    wandb.config.update(cfg)
    args = wandb.config
    cudnn.enabled = True

    if args.experiment is not None:
        api = wandb.Api()
        run = api.run(path=f'my_user/FGSM/{wandb.run.id}')
        run.name = f'{args.experiment}-{run.name}'
        run.save()

    code_path = os.path.join(os.path.join(wandb.run.dir, 'files', 'full_code'))
    Path(code_path).mkdir(parents=True, exist_ok=True)
    for file in glob.glob('*.py'):
        shutil.copy(file, code_path)
    # shutil.copytree('configs', code_path + '/configs', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    output_path = wandb.run.dir.replace('/wandb/', '/outputs/')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_path)

    train_dataset = ThreeClassTensorDataset(root='.',
                                      data_files='captions_gt_features.pickle',
                                      p=args.p,
                                      feature_model=args.feature_model, k=args.k,
                                        dataset=args.dataset, split='supervised_train', fastdebug=args.fastdebug)

    test_dataset = ImageTensorDataset(root='.', data_files=['captions_prediction_trainval_sup_sat_features.pickle', 'captions_prediction_trainval_sup_aoanet_features.pickle'],
                                      feature_model=args.feature_model, dataset=args.dataset, split='supervised_test', fastdebug=args.fastdebug)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model = SiameseFCN(shared_in_features=1024, shared_out_feature=32, num_labels=3).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    corpus_name = ('aab' if args.dataset == 'CUB' else 'wiki') if args.corpus is None else args.corpus
    criterion = CrossEntropyLoss('.', model, dataset=args.dataset, corpus_name=corpus_name, loss_type=args.loss_type,
                                     feature_model=args.feature_model, loss_mult=args.loss_mult, three=args.three).to(device)
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
    for epoch in trange(start_epoch, args.epochs, desc="Epoch"):
        model.train()
        loss_epoch = 0
        for training_steps, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            loss_value = criterion(features, labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_epoch += loss_value.item()
            if writer is not None:
                writer.add_scalar('training_statistics/loss', loss_value.item(),
                                  epoch * len(train_dataloader) + training_steps)
                if training_steps % 100 == 0:
                    n = epoch * len(train_dataloader) + training_steps
                    for pi, p in enumerate(optimizer.param_groups):
                        writer.add_scalar(f'training_statistics/learning_rate_group{pi}', p['lr'], n)
                    if training_steps == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch+1)
                            if value.grad is not None:
                                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch+1)

        wandb.log({'train/loss': loss_epoch / len(train_dataloader)}, step=epoch)
        if (epoch + 1) % args.evaluation_steps == 0:
            model.eval()
            with torch.no_grad():
                predictions = []
                ground_truths = []
                for eval_steps, (features, labels) in enumerate(tqdm(test_dataloader)):
                    features, labels = features.to(device), labels.to(device)
                    scores = criterion(features, return_scores=True)
                    predictions.append(scores)
                    ground_truths.append(labels)
                predictions = torch.cat(predictions, 0)
                ground_truths = torch.cat(ground_truths, 0)
                acc = accuracy(predictions, ground_truths, test_dataloader.dataset.present_classes, topk=(1, 5))
                mrank = rank(predictions, ground_truths, test_dataloader.dataset.present_classes)
                logging.info(f'Accuracy @1: {acc[0]} @5: {acc[1]} mRank: {mrank}')
                print(f'Accuracy @1: {acc[0]} @5: {acc[1]} mRank: {mrank}')
                if writer is not None:
                    writer.add_scalar('evaluation_document/supervised_test_gzsl/top1', acc[0], epoch+1)
                    writer.add_scalar('evaluation_document/supervised_test_gzsl/top5', acc[1], epoch+1)
                    writer.add_scalar('evaluation_document/supervised_test_gzsl/mean_rank', mrank, epoch+1)
                    wandb.log({'test/top1': acc[0], 'test/top5': acc[1], 'test/mean_rank': mrank}, step=(epoch + 1))
            if output_path is not None:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                # Save the state
                torch.save(state, f'{output_path}/checkpoint.{epoch+1}.pth')


if __name__ == '__main__':

    args = get_args()
    args.three = True
    if args.load_checkpoint is not None:
        with open(os.path.dirname(args.load_checkpoint)+'/logs/args.json', 'rb') as f:
            temp = json.load(f)
        del temp['experiment']
        del temp['gpu']
        del temp['evaluation_steps']
        del temp['fastdebug']
        del temp['num_workers']
        args = args.__dict__
        args.update(**temp)
        args = SimpleNamespace(**args)
    main(args)
