import itertools
import json
import os.path
import pickle

import sys

import nltk
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm import tqdm


from modules.models import SiameseSentenceTransformer
import logging
import en_core_web_lg

nlp = en_core_web_lg.load()


class CUBDatasetEvaluator(SentenceEvaluator):

    def __init__(self, root='.', split='test_unseen', type='zsl', sbert_evaluators=[],  name='', writer=None, args=None, all_scores=False, json_path=None, noun_vocab=None, validation_evaluator=False, chunk_n=-1):
        self.args = args

        self.name = name
        self.writer = writer
        self.sbert_evaluators = sbert_evaluators
        self.split = split
        self.type = type
        self.noun_vocab = noun_vocab
        self.eval_method = self.args['eval_method']
        self.all_scores = all_scores
        self.json_path = json_path
        self.dataset = args['dataset']
        self.dataset_lower = self.dataset.lower()
        self.corpus_name = self.args['eval_corpus']
        self.best_scores = None
        self.chunk_n = chunk_n
        if self.json_path is None:
            self.json_filename = None
        else:
            self.json_filename = os.path.join(self.json_path, f'{self.split}_{self.type}.json')

        if validation_evaluator:
            self.validation_key = f"{self.split}_{self.type}/top5"
        else:
            self.validation_key = None

        with open(root + f'/{self.dataset}/image_data.json', 'r') as handle:
            self.image_data = json.load(handle)
        self.image_metadata = [self.image_data['images'][i] for i in self.image_data[self.split + '_loc']]
        if self.type == 'zsl':
            self.classes = sorted(list({v['class_name'] for v in self.image_metadata}))
            self.present_classes = list(range(len(self.classes)))
        else:
            self.classes = sorted(list({v['class_name'] for v in self.image_data['images']}))
            self.present_classes = {v['class_name'] for v in self.image_metadata}
            self.present_classes = [self.classes.index(c) for c in self.present_classes]

        self.nclass = len(self.classes)

        self.labels = torch.tensor([self.classes.index(v['class_name']) for v in self.image_metadata])

        with open(root + f'/{self.dataset}/corpus_{self.corpus_name}.pickle', 'rb') as handle:
            self.class_corpus = pickle.load(handle)

        self.query_chunks = {}
        self.eval_data = self.args['eval_data']
        logging.info("Eval nethod: " + self.eval_data)
        logging.info(f"#{self.split} classes: {self.nclass}")

        if self.eval_data in ['captions_gt', 'captions_pred']:
            if self.eval_data == 'captions_gt' or self.eval_data == 'captions_gt_viz':
                with open(root + f'/{self.dataset}/captions_gt.pickle', 'rb') as handle:
                    captions_gt = pickle.load(handle)
            else:
                pred_source = self.args['pred_source']
                pred_sources = pred_source.split(',')
                logging.info(f'Using captions from /{self.dataset}/captions_prediction_trainval_{pred_sources[0]}.pickle')
                with open(root + f'/{self.dataset}/captions_prediction_trainval_{pred_sources[0]}.pickle', 'rb') as handle:
                    captions_pred = pickle.load(handle)
                if len(pred_sources) > 1:
                    for pred_source_ in pred_sources[1:]:
                        logging.info(f'Using captions from /{self.dataset}/captions_prediction_trainval_{pred_source_}.pickle')
                        with open(root + f'/{self.dataset}/captions_prediction_trainval_{pred_source_}.pickle', 'rb') as handle:
                            captions_pred2 = pickle.load(handle)
                        for key, value in captions_pred.items():
                            captions_pred[key] = captions_pred2[key]+value

            if self.eval_data in ['captions_gt', 'captions_gt_viz']:
                self.query_chunks = captions_gt
            else:
                self.query_chunks = captions_pred

        else:
            logging.error(self.eval_data + ' is a wrong eval method')
            sys.exit(1)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, num_batches=1, best_scores=None, two_c=True, thrsh=None):
        if epoch % 3 == 0:
            if self.sbert_evaluators is not None and len(self.sbert_evaluators) > 0:
                for sbert_evaluator in self.sbert_evaluators:
                    sbert_evaluator(model, output_path, epoch, steps, num_batches=num_batches)

        model.eval()

        scores, ranks = self.evaluateF(model, two_c=two_c, thrsh=thrsh)
        logging.info(f"Accuracy on {'visualizer' if self.json_path is not None else ''} {self.dataset} {self.split} {self.type.upper()} : Top@1 {scores[0]:.2f} Top@5 {scores[1]:.2f} mRank {ranks:.2f}\n")
        score_dicts = {f'evaluation_document/{self.split}_{self.type}/top1': scores[0], f'evaluation_document/{self.split}_{self.type}/top5': scores[1], f'evaluation_document/{self.split}_{self.type}/mean_rank': ranks}

        if self.writer is not None:
            if epoch == -1:
                n = 0
            else:
                n = (epoch + 1) if steps == -1 else epoch + steps / num_batches
                n = int(n * 1000)  # Tensorboard does not support fractional steps

            for k, v in score_dicts.items():
                self.writer.add_scalar(k, v, n)

        if self.all_scores:
            return score_dicts
        else:
            return scores[1]

    @staticmethod
    def tokenlist_to_str(tokenlist):
        return " ".join([t.text for t in tokenlist]).replace(' - ', '-')

    def get_embeddings(self, model, classes):
        return self.get_doc_embeddings(model, classes), self.get_attr_embeddings(model, classes)

    def get_doc_sentences(self, model, classes):
        document_list = self.preprocess_docs(classes)
        return document_list

    def get_doc_embeddings(self, model, classes):
        document_list = self.preprocess_docs(classes)
        document_embeddings = [self.get_prediction(sents, model) for sents in document_list]
        return document_embeddings

    def get_attr_embeddings(self, model, classes):
        attribute_list = [self.present_attrs[cls] for cls in classes]
        attributes_scores = self.get_prediction(self.attr_list, model)
        attribute_embeddings = [attributes_scores[attr_present] for attr_present in attribute_list]
        return attribute_embeddings

    def get_sentences(self, model, classes):
        attribute_list = [[self.attr_list[i] for i in self.present_attrs[cls]] for cls in classes]
        document_list = self.preprocess_docs(classes)
        return attribute_list, document_list

    def evaluateF(self, model, images_subset=None, return_scores=False, two_c=False, thrsh=None):
        two_c = True
        logging.info(f"two_c {two_c} thrsh {thrsh}")
        logging.info(f"two_c {two_c} thrsh {thrsh}")
        if self.eval_method == 'classification' or self.eval_method == 'cosine':
            document_embeddings = self.get_doc_embeddings(model.get_sbert(), self.classes)
            if 'caption' not in self.eval_data:
                attributes_scores = self.get_prediction(self.attr_list, model.get_sbert())
        elif self.eval_method == 'concat':
            document_embeddings = self.get_doc_sentences(model.get_sbert(), self.classes)
        doc_scores = torch.ones(len(self.image_metadata), len(document_embeddings))
        score_detail = []
        if images_subset is None:
            images_subset = self.image_metadata
        for ai, item in enumerate(tqdm(images_subset)):

            details = []
            if self.eval_method == 'concat':
                if self.eval_data in ['captions_gt', 'captions_pred', 'captions_gt_viz', 'captions_pred_viz']:
                    attributes = self.query_chunks[item['id']]
                elif self.eval_data == 'attr_gt_image' or self.eval_data == 'attr_pred_image':
                    attributes = [self.attr_list[i] for i in self.query_chunks[item['id']].nonzero()[0]]
            else:
                if self.eval_data in ['captions_gt', 'captions_pred', 'captions_gt_viz', 'captions_pred_viz']:
                    attributes = self.get_prediction(self.query_chunks[item['id']], model.get_sbert())
                elif self.eval_data == 'attr_gt_image' or self.eval_data == 'attr_pred_image':
                    attributes = attributes_scores[self.query_chunks[item['id']].nonzero()[0]]
            for di, sentences in enumerate(document_embeddings):
                if self.eval_method == 'cosine':
                    score = self.cosine_similarity(sentences, attributes)
                    doc_scores[ai, di] = score.max(0).values.mean(0)
                    if self.json_filename is not None:
                        score_f = torch.zeros_like(score)
                        score_max, score_argmax = score_f.max(0)
                        score_f[score_argmax] = score_max
                        details.append((1000 * torch.cat([score_f.unsqueeze(2), score.unsqueeze(2), 1 - score.unsqueeze(2), 0 * score.unsqueeze(2)], 2).permute(1, 0, 2)).cpu().numpy().astype(int).tolist())
                else:
                    if self.eval_method == 'classification':
                        score_mlp = self.classifier_on_embeddings_product(model, sentences, attributes)
                    elif self.eval_method == 'concat':
                        score_mlp = self.classifier_on_sentences_product(model, sentences, attributes)
                    if isinstance(score_mlp, tuple):
                        score_mlp, nouns = score_mlp
                    else:
                        nouns = None
                    score, score_argmax = score_mlp[:, :, :2].max(2)
                    score_weight = torch.zeros_like(score_argmax).float()
                    score_weight[score_argmax == 0] = -1.  # label_neg
                    score_weight[score_argmax == 1] = 1.  # label_pos
                    score_weight[score_argmax == 2] = 0.  # label_neut

                    score = score * score_weight
                    doc_scores[ai, di] = score.mean(0).mean(0)

        if return_scores:
            return {"scores": (doc_scores * 1000).cpu().numpy().astype(int).tolist(), "details": score_detail}

        return self.accuracy(doc_scores, self.labels, self.present_classes, None, topk=(1, 5)), self.rank(doc_scores, self.labels, self.present_classes)

    def preprocess_docs(self, var):
        docs = [nltk.sent_tokenize(self.class_corpus[cls]['text']) for cls in var]
        docs = [[self.preprocess(sent) for sent in sents] for sents in docs]
        return docs

    @staticmethod
    def cut_dataset(dataset, n):
        datset_small = []
        all_classes = set(item['class_name'] for item in dataset)
        count_dict = dict(zip(list(all_classes), [0] * len(all_classes)))
        for item in tqdm(dataset):
            if count_dict[item['class_name']] < n:
                datset_small.append(item)
                count_dict[item['class_name']] += 1
        return datset_small

    @staticmethod
    def create_or_append(dict_of_list, key, value):
        if key in dict_of_list:
            dict_of_list[key].append(value)
        else:
            dict_of_list[key] = [value]

    def classifier_on_embeddings_product(self, model: SiameseSentenceTransformer, feat_a, feat_b):
        with torch.no_grad():
            output = model.get_classifier_scores(torch.repeat_interleave(feat_a, repeats=feat_b.size(0), dim=0), feat_b.repeat(feat_a.size(0), 1))
            return F.softmax(output.view(len(feat_a), len(feat_b), -1), dim=-1)

    @staticmethod
    def classifier_on_sentences_product(model: SiameseSentenceTransformer, sents_a, sents_b):
        texts = []
        if hasattr(model._first_module().tokenizer, 'sep_token'):
            sep_token = model._first_module().tokenizer.sep_token
        else:
            sep_token = '[PADDING_TOKEN]'
        for a, b in itertools.product(sents_a, sents_b):
            texts.append(a + '  ' + sep_token + ' ' + b)
        output = CUBDatasetEvaluator.get_prediction(texts, model, 64)
        return F.softmax(output.view(len(sents_a), len(sents_b), -1), dim=-1)

    @staticmethod
    def get_prediction(lines, model: SentenceTransformer,  batch_size=16):
        with torch.no_grad():
            return torch.stack(model.encode(lines, convert_to_numpy=False, batch_size=batch_size, show_progress_bar=False), 0)

    @staticmethod
    def preprocess(string):
        string = string.replace('\n', ' ')
        # string = re.sub(r"[^A-Za-z0-9(),\.!?\'\`]", " ", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\.", " . ", string)
        # string = re.sub(r"\(", " ( ", string)
        # string = re.sub(r"\)", " ) ", string)
        # string = re.sub(r"\?", " ? ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    @staticmethod
    def accuracy(output, target, present_classes, super_map = None, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)

            _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            if super_map is not None:
                pred = super_map[pred]
                target = super_map[target]
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].float().sum(0)
                res.append(CUBDatasetEvaluator.compute_per_class_metric(correct_k, target, present_classes))
            return [r.item() * 100. for r in res]

    @staticmethod
    def rank(output, target, present_classes, super_map = None):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            pred = output.argsort(dim=1, descending=True)
            if super_map is not None:
                pred = super_map[pred]
                target = super_map[target]
            rank = pred.eq(target.view(-1, 1).expand_as(pred)).nonzero(as_tuple=False)[:, 1].float()
            return CUBDatasetEvaluator.compute_per_class_metric(rank + 1, target, present_classes).item()

    @staticmethod
    def compute_per_class_metric(metric, target, present_classes):
        acc_per_class = 0.
        for i in present_classes:
            idx = (target == i)
            e = torch.true_divide(torch.sum(metric[idx]), torch.sum(idx))
            acc_per_class += e
        return acc_per_class / len(present_classes)

    @staticmethod
    def cosine_similarity(x1, x2=None):
        x1 = F.normalize(x1)
        x2 = x1 if x2 is None else F.normalize(x2)
        return torch.mm(x1, x2.t())
