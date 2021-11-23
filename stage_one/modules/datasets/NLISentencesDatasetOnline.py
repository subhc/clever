import random
import re
import sys
from types import SimpleNamespace

import os
from nlp import load_dataset
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.readers import InputExample
from typing import List
import torch
import logging
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from tqdm import tqdm

import random

import numpy as np
import json
import nltk
import pickle
from nltk.corpus import wordnet
import en_core_web_lg
from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB, AUX, PRON, PROPN
from spacy.symbols import amod, acomp, nsubj, attr, advmod, npadvmod
from tqdm import tqdm
import json


nlp = en_core_web_lg.load()

label_neg = 0
label_pos = 1
label_neut = 2

parts = ['back', 'beak', 'bill', 'belly', 'breast', 'crown', 'underparts', 'forehead', 'eye', 'leg', 'wing', 'nape', 'tail', 'throat']
partsed = ['backed', 'beaked', 'billed', 'bellied', 'breasted', 'crowned', 'eyed', 'legged', 'winged', 'naped', 'tailed', 'throated']
parts_set = set(parts)

class NLISentencesDatasetOnline(Dataset):
    wiki = None

    def __init__(self, root, split: str, model, model_name, train_data: str, mode='nli', fastdebug=False, dataset='CUB', eval_method='classification', robust_n=1, noun_vocab=None):
        logging.info(f"Dataset: {dataset} Class Split: {split}  Training Data: {train_data}")
        if robust_n > 1:
            logging.info(f"Robust N: {robust_n}")
        self.fastdebug = fastdebug
        self.eval_method = eval_method
        self.dataset = dataset
        self.train_data = train_data
        self.split = split
        self.mode = mode
        self.model_name = model_name
        self.robust_n = robust_n
        self.model = model
        if hasattr(model, 'sbert_model'):
            self.tokenizer = model.get_sbert()._first_module().tokenizer
        else:
            self.tokenizer = model._first_module().tokenizer
        if hasattr(self.tokenizer, 'sep_token'):
            self.sep_token = [self.tokenizer.sep_token_id]
        else:
            self.sep_token = [torch.zeros(300)]
        self.classes = ['all']

        with open(root + f'/{self.dataset}/image_data.json', 'r') as handle:
            self.image_data = json.load(handle)
        self.image_metadata = [self.image_data['images'][i] for i in self.image_data[self.split + '_loc']]
        if self.fastdebug:
            self.image_metadata = random.sample(self.image_metadata, 200)
        logging.info(f"No. of {split} images in {self.dataset}: {len(self.image_metadata)}")
        logging.info(f"No. of {split} classes in {self.dataset}: {len({v['class_name'] for v in self.image_metadata})}")
        self.len = 100 if self.fastdebug else len(self.image_metadata)*10  # #captions

        self.impute_chunk_synant = False
        self.sent_dict_id = {}
        self.data_dict_cls = {}
        self.log_count = 0

        self.pos_dict = {}
        self.neg_dict = {}
        self.neut_dict = {}

        self.noun_dict_map = {}
        self.noun_vocab = noun_vocab
        self.noun_vocab_index_map = None
        self.noun_vocab_len = -1

        self.tokens = None
        self.labels = None

        tensor_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        self.labels = tensor_labels
        self.random_neut_list = None

        self.convert_input_examples(root)

    def get_noun_vocab_size(self):
        return self.noun_vocab_len

    def get_noun_vocab(self):
        return self.noun_vocab

    def convert_input_examples(self, root):


        sent_dict_id = {}
        sent_dict_cls = {}
        if self.train_data == 'aab':
            with open(root + f'{self.dataset}/corpus_aab.pickle', 'rb') as handle:
                data = pickle.load(handle)
            for cls in self.classes:
                sent_dict_cls[cls] = nltk.sent_tokenize(data[cls]['text'])
                sent_dict_id[cls] = [sent_dict_cls[cls]]
        elif self.train_data == 'wiki':
            with open(root + f'{self.dataset}/corpus_wiki.pickle', 'rb') as handle:
                data = pickle.load(handle)
            for cls in self.classes:
                sent_dict_cls[cls] = nltk.sent_tokenize(data[cls]['text'])
                sent_dict_id[cls] = [sent_dict_cls[cls]]

        elif self.train_data == 'captions_gt':

            with open(root + f'/{self.dataset}/captions_gt.pickle', 'rb') as handle:
                captions_gt = pickle.load(handle)

            for item in self.image_metadata:
                sents = set([str(t).replace('pedal', 'petal') for t in captions_gt[item['id']]])
                if 'all' in sent_dict_cls:
                    sent_dict_cls['all'].update(sents)
                else:
                    sent_dict_cls['all'] = sents

            for item in self.image_metadata:
                sents = list({str(t).replace('pedal', 'petal') for t in captions_gt[item['id']]})
                if 'all' in sent_dict_id:
                    sent_dict_id['all'].append(sents)
                else:
                    sent_dict_id['all'] = [sents]


        elif self.train_data == 'captions_pred':

            with open(root + f'/{self.dataset}/captions_bs10_agnostic_prediction.pickle', 'rb') as handle:
                captions_pred = pickle.load(handle)


            for item in self.image_metadata:
                sents = set([str(t) for t in captions_pred[item['id']]])
                if 'all' in sent_dict_cls:
                    sent_dict_cls['all'].update(sents)
                else:
                    sent_dict_cls['all'] = sents

            for item in self.image_metadata:
                sents = list(set([str(t) for t in captions_pred[item['id']]]))
                if 'all' in sent_dict_id:
                    sent_dict_id['all'].append(sents)
                else:
                    sent_dict_id['all'] = [sents]
        else:
            logging.error("Wrong split: " + self.split)
            sys.exit(1)

        self.sent_dict_id = sent_dict_id
        self.sent_dict_cls = sent_dict_cls

        logging.info(f"Tokenizing the sent_dicts with {self.tokenizer.__class__.__name__}")

    def __getitem__(self, item):
        pcls = str(random.choice(self.classes))
        chance = random.random()
        if chance < 0.5:  # -------------------- Positive --------------------
            label = 1
            pos_caps = random.choice(range(len(self.sent_dict_id[pcls])))
            pos_caps_a = random.choice(range(len(self.sent_dict_id[pcls][pos_caps])))
            pos_caps_p = random.choice(range(len(self.sent_dict_id[pcls][pos_caps])))
            sents = [self.sent_dict_id[pcls][pos_caps][pos_caps_a], self.sent_dict_id[pcls][pos_caps][pos_caps_p]]

        else:             # -------------------- Negative --------------------
            label = 0
            img_i = random.choice(range(len(self.sent_dict_id[pcls])))
            cap_i = random.choice(range(len(self.sent_dict_id[pcls][img_i])))
            ncls = 'all'
            ni_i = random.choice([x for x in range(len(self.sent_dict_id[ncls])) if x != img_i])
            nc_i = random.choice(range(len(self.sent_dict_id[pcls][ni_i])))
            sents = [self.sent_dict_id[pcls][img_i][cap_i], self.sent_dict_id[ncls][ni_i][nc_i]]
        return [self.model.tokenize(sent) for sent in sents], torch.scalar_tensor(label).long(), torch.scalar_tensor(0).long(), ["dummy"], torch.FloatTensor([-1, -1, -1])

    def __len__(self):
        return self.len
