import logging
import os
from pathlib import Path
from typing import Tuple, Iterable, Dict, List
from typing import Type

import json

import copy
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from torch.utils.tensorboard.summary import hparams


class SentenceTransformerCustom(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super(SentenceTransformerCustom, self).__init__(*args, **kwargs)
        self.best_scores = {}
        self.training_states = {}
        self.register_trainable_modules()

    def register_trainable_modules(self):
        for key, m in self.named_modules():
            self.training_states[key] = m.training

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_epochs: int = 1,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            lr: Dict[str, float] = {'encoder': 2e-6, 'fc': 2e-5},
            weight_decay: Dict[str, float] = {'encoder': .01, 'fc': 5e-5},
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1,
            task='triplet',
            writer: SummaryWriter = None,
            aux_lr_mult=1.,
            args=None
            ):
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if [f for f in os.listdir(output_path) if f != 'logs']:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self.device

        for loss_model in loss_models:
            loss_model.to(device)


        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            param_optimizer = [p for p in param_optimizer if p[1].requires_grad]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            sbert_indentifier = ".".join(list(loss_model.model.get_sbert().named_parameters())[0][0].split('.')[:2])  # 0.bert or 0.roberta
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if (sbert_indentifier in n) and not any(nd in n for nd in no_decay)], 'lr': lr['encoder'], 'weight_decay': weight_decay['encoder'], 'correct_bias': False},
                {'params': [p for n, p in param_optimizer if (sbert_indentifier in n) and any(nd in n for nd in no_decay)], 'lr': lr['encoder'], 'weight_decay': 0.0, 'correct_bias': False},
                {'params': [p for n, p in param_optimizer if sbert_indentifier not in n and not any(nd in n for nd in no_decay)], 'lr': lr['fc'], 'weight_decay': weight_decay['fc']},
                {'params': [p for n, p in param_optimizer if sbert_indentifier not in n and any(nd in n for nd in no_decay)], 'lr': lr['fc'], 'weight_decay': 0.0},
            ]
            optimizer_grouped_parameters = [p for p in optimizer_grouped_parameters if p['params']]  # remove empty groups

            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            if scheduler.lower() == 'warmupcosinewithhardrestarts':
                scheduler_obj = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * warmup_epochs, num_training_steps=t_total, num_cycles=epochs // 2)
            else:
                scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=steps_per_epoch * warmup_epochs, t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(loss_models)):
                model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx],
                                                  opt_level=fp16_opt_level)
                loss_models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)
        n_iter = 0
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()
            counts = 0
            label_counts = torch.zeros(2).cuda()
            example_types_counts = torch.zeros(4).cuda()
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels, example_types, nouns, counts = self.batch_to_device(data, self.device)
                    loss_value = loss_model(features, labels)
                    label_counts += torch.bincount(labels, minlength=label_counts.size(0))
                    if writer is not None:
                        if training_steps % 100:
                            n = (epoch + 1) if training_steps == -1 else epoch + training_steps / len(dataloaders[train_idx].batch_sampler)
                            n = int(n * 1000)  # Tensorboard does not support fractional steps
                            writer.add_scalar('training_statistics/loss', loss_value.item(), n)
                            for pi, p in enumerate(optimizer.param_groups):
                                writer.add_scalar(f'training_statistics/learning_rate_group{pi}', p['lr'], n)

                    if fp16:
                        with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                training_steps += 1
                global_step += 1

            if writer is not None:
                if task == 'triplet':
                    writer.add_scalar('training_statistics/nepoch_frac_active', counts * 1.0 / len(dataloaders[0].dataset), epoch)
                elif task == 'nli':
                    label_counts = label_counts.cpu().numpy()
                    logging.info("Examples seen: positive: {:.2f} negative: {:.2f}".format(100. * label_counts[1] / sum(label_counts), 100. * label_counts[0] / sum(label_counts)))

            if epoch % evaluation_steps == 0:
                self.eval_during_training(args, evaluator, writer, output_path, save_best_model, epoch, -1)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        nouns = []
        counts = []
        example_types = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label, example_type, noun, count in batch:
            labels.append(label)
            example_types.append(example_type)
            nouns.append(noun)
            counts.append(count)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])

            for feature_name in feature_lists:
                # feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
                feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)
        return {'features': features, 'labels': torch.stack(labels), 'example_types': torch.stack(example_types), 'nouns': nouns, 'counts': torch.stack(counts)}

    def get_sbert(self):
        return self

    def train(self, mode: True):
        for key, m in self.named_modules():
            m.training = self.training_states[key]
        return self

    def eval(self):
        for key, m in self.named_modules():
            m.training = False
        return self

    def eval_during_training(self, args, evaluators, writer, output_path, save_model, epoch, steps, num_batches=1):
        """Runs evaluation during the training"""
        if output_path is not None:
            if steps == -1 and epoch > -1 and save_model:
                Path(output_path + '/{}'.format(int(epoch + 1))).mkdir(parents=True, exist_ok=True)
                self.save(output_path + '/{}'.format(int(epoch + 1)))
        viz_evaluator = None
        validation_key = None
        n = (epoch + 1) if steps == -1 else epoch + steps / num_batches
        n = int(n * 1000)  # Tensorboard does not support fractional steps
        if evaluators is not None and len(evaluators) > 0:
            scores = {}
            for evaluator in evaluators:
                if evaluator.validation_key is not None:
                    validation_key = evaluator.validation_key
                if evaluator.json_path is not None:
                    viz_evaluator = evaluator
                else:
                    score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps, num_batches=num_batches)
                    scores.update(score)

        msg = "Accuracy on  CUB overall GZSL : "
        nmsg = len(msg)
        for metric, metric_name in zip(['top1', 'top5', 'mean_rank'], ['Top@1', 'Top@5', 'mRank']):
            if f"evaluation_document/test_unseen_gzsl/{metric}" in scores and f"evaluation_document/test_seen_gzsl/{metric}" in scores:
                v = 2*(scores[f"evaluation_document/test_seen_gzsl/{metric}"]*scores[f"evaluation_document/test_unseen_gzsl/{metric}"])/(scores[f"evaluation_document/test_seen_gzsl/{metric}"]+scores[f"evaluation_document/test_unseen_gzsl/{metric}"]+1e-5)
                scores[f"evaluation_document/test_overall_gzsl/{metric}"] = v
                msg += f'{metric_name} {v:.2f} '
                if writer is not None:
                    writer.add_scalar(f"evaluation_document/test_overall_gzsl/{metric}", v, n)
        if len(msg) > nmsg:
            logging.info(msg)
        if validation_key is None and "evaluation_document/test_overall_gzsl/top5" in scores:
            validation_key = 'test_overall_gzsl/top5'
        if output_path is not None:

            if len(scores) > 0:
                if os.path.exists(f'{output_path}/{epoch + 1}') and steps == -1:
                    with open(f'{output_path}/{epoch + 1}/results.json', 'w') as outfile:
                        json.dump({'epoch': epoch + 1, 'steps': steps, 'scores': scores, 'best_scores': self.best_scores}, outfile)
                if validation_key is not None and (len(self.best_scores) == 0 or
                   scores['evaluation_document/'+validation_key] >= self.best_scores['validation_best/'+validation_key]):

                    for key, value in scores.items():
                        self.best_scores[key.replace('evaluation_document', 'validation_best')] = value
                    self.best_scores['epoch'] = epoch

                    if save_model:
                        Path(output_path + '/best').mkdir(parents=True, exist_ok=True)
                        self.save(output_path + '/best')
                        print(self.best_scores)
                        with open(f'{output_path}/best/results.json', 'w') as outfile:
                            json.dump({'epoch': epoch + 1, 'steps': steps, 'scores': scores, 'best_scores': self.best_scores}, outfile)

                    best_scores1 = copy.deepcopy(self.best_scores)
                    del best_scores1['epoch']
                    self.add_hparams(writer, args, best_scores1, n)

                    if viz_evaluator is not None:
                        logging.info("Generating visualization")
                        viz_evaluator(self, output_path=None, epoch=epoch, steps=steps, num_batches=num_batches, best_scores=scores)
        return self.best_scores


    def add_hparams(self, writer, hparam_dict=None, metric_dict=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            writer.add_scalar(k, v, global_step)

    @staticmethod
    def batch_to_device(batch, target_device: torch.device):
        """
        send a batch to a device

        :param batch:
        :param target_device:
        :return: the batch sent to the device
        """
        features = batch['features']
        for paired_sentence_idx in range(len(features)):
            for feature_name in features[paired_sentence_idx]:
                features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)

        labels = batch['labels'].to(target_device)
        example_types = batch['example_types'].to(target_device)
        nouns = batch['nouns']
        counts = batch['counts'].to(target_device)
        return features, labels, example_types, nouns, counts

    @staticmethod
    def load(path):
        return SentenceTransformerCustom(path)
