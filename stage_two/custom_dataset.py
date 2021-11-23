import json
import os
import pickle
import random
import logging
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import os.path as osp

from tqdm import tqdm

import en_core_web_lg


def get_features(root, dataset, feature_model, data_files):
    if isinstance(data_files, list):
        feature_location = f'{root}/{dataset}/{feature_model}/{data_files[0]}'
        with open(feature_location, 'rb') as f:
            precomputed_features_dict = pickle.load(f)
        for data_file in data_files[1:]:
            feature_location = f'{root}/{dataset}/{feature_model}/{data_file}'
            with open(feature_location, 'rb') as f:
                precomputed_features_dict1 = pickle.load(f)
            for key, value in precomputed_features_dict.items():
                if key in precomputed_features_dict1:
                    precomputed_features_dict[key] = torch.cat((precomputed_features_dict1[key], value), 0)
    else:
        feature_location = f'{root}/{dataset}/{feature_model}/{data_files}'
        with open(feature_location, 'rb') as f:
            precomputed_features_dict = pickle.load(f)
    return precomputed_features_dict


class ImageTensorDataset(Dataset):
    def __init__(self, root, data_files, feature_model, dataset, split, fastdebug=False):
        super(ImageTensorDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        self.split = split
        self.feature_model = feature_model
        self.fastdebug = fastdebug
        with open(root + f'/{dataset}/image_data.json', 'r') as handle:
            self.image_data = json.load(handle)
        print(self.split)
        if self.split == 'supervised_train':
            indices = self.image_data['supervised_train_loc']
            indices.extend(self.image_data['supervised_val_loc'])
        elif self.split == 'supervised_test':
            indices = self.image_data['supervised_test_loc']

        # if split == 'supervised_test':
        #     indices = self.image_data['supervised_val_loc']
        self.image_metadata = [self.image_data['images'][i] for i in indices]

        self.classes = sorted(list({v['class_name'] for v in self.image_data['images']}))
        self.present_classes = list(range(len(self.classes)))

        precomputed_features_dict = get_features(root, dataset, feature_model, data_files)
        self.precomputed_features = [precomputed_features_dict[item['id']] for item in self.image_metadata]
        self.labels = [self.classes.index(item['class_name']) for item in self.image_metadata]

    def __getitem__(self, index):
        return self.precomputed_features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class PosNegTensorDataset(ImageTensorDataset):
    def __init__(self, k=0, p=0, *args, **kw):
        super(PosNegTensorDataset, self).__init__(*args, **kw)

        # with open(f'./CUB/captions_gt_indices_word.json', 'r') as handle:
        #     good_ones = json.load(handle)
        # ids = [item['id'] for item in self.image_metadata]
        # self.precomputed_features = [self.precomputed_features[i][good_ones[id]] for i, id in enumerate(ids)]

    def __getitem__(self, index):
        chance = random.random()
        if chance < 1.0 / 2:
            label = 1
            pos_caps = random.choice(range(len(self.precomputed_features)))
            pos_caps_a, pos_caps_p = random.sample(range(self.precomputed_features[pos_caps].size(0)), k=2)
            sents = torch.stack([self.precomputed_features[pos_caps][pos_caps_a], self.precomputed_features[pos_caps][pos_caps_p]], dim=0)
        else:
            label = 0
            pos_caps = random.choice(range(len(self.precomputed_features)))
            neg_caps = (random.choice(range(len(self.precomputed_features) - 1)) + pos_caps + 1) % len(self.precomputed_features)
            assert pos_caps != neg_caps
            pos_caps_a = random.choice(range(self.precomputed_features[pos_caps].size(0)))
            neg_caps_p = random.choice(range(self.precomputed_features[neg_caps].size(0)))
            sents = torch.stack([self.precomputed_features[pos_caps][pos_caps_a], self.precomputed_features[neg_caps][neg_caps_p]], dim=0)
        return sents, label

    def __len__(self):
        return len(self.labels)

class ThreeClassTensorDataset(ImageTensorDataset):
    def __init__(self, k=0, rewrite_cache=False, train_data='roberta', p=0, *args, **kw):
        super(ThreeClassTensorDataset, self).__init__(*args, **kw)
        self.rewrite_cache = rewrite_cache
        self.train_data = train_data
        self.log_count = 0
        self.p = p

        self.neg_dict = defaultdict(list)
        self.neut_dict = defaultdict(list)
        self.neut_dict_corpus = defaultdict(list)
        if self.fastdebug:
            self.image_metadata = random.sample(self.image_metadata, 20)

        with open(self.root + f'/{self.dataset}/corpus_{"aab_noname" if self.dataset == "CUB" else "wiki"}.pickle', 'rb') as handle:
            corpus = pickle.load(handle)

        with open(self.root + f'/{self.dataset}/FGSM/corpus_{"aab_noname" if self.dataset == "CUB" else "wiki"}_features.pickle', 'rb') as handle:
            corpus_features = pickle.load(handle)

        self.corpus_sents = [text for texts in corpus.values() for text in texts['sents']]
        self.precomputed_features_corpus = torch.cat([feats for feats in corpus_features.values()], 0)

        self.convert_input_examples()

    def convert_input_examples(self):

        nlp = en_core_web_lg.load()
        self.sent_dict_id = []
        cache_filename = self.root + f'/cache/{self.dataset}/sent_dict_{self.dataset}_{self.train_data}_{self.split}{"_fastdebug" if self.fastdebug else ""}.pickle'
        if os.path.exists(cache_filename) and not self.rewrite_cache:
            with open(cache_filename, 'rb') as handle:
                self.sent_dict_id = pickle.load(handle)
                logging.info(f'Loaded {cache_filename} from cache')
        else:

            with open(self.root + f'/{self.dataset}/captions_gt.pickle', 'rb') as handle:
                captions_gt = pickle.load(handle)

            for item in self.image_metadata:
                self.sent_dict_id.append(captions_gt[item['id']])

            logging.info(f"Saving into cache: {cache_filename}")
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            pickle.dump(self.sent_dict_id, open(cache_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        cache_filename = self.root + f'/cache/{self.dataset}/noun_pos_neg_neut_{self.dataset}_{self.train_data}_{self.split}{"_fastdebug" if self.fastdebug else ""}.pickle'
        if os.path.exists(cache_filename) and not self.rewrite_cache:  # and 'debug' not in self.split
            with open(cache_filename, 'rb') as handle:
                logging.info(f'Loading {cache_filename} from cache')
                self.neg_dict, self.neut_dict, self.neut_dict_corpus = pickle.load(handle)
                logging.info(f'Loaded {cache_filename} from cache')
        else:
            logging.info(f"Preprocessing with spacy...")
            print(f"Preprocessing with spacy...")
            noun_dict_unfiltered = {}
            noun_dict = {}
            sent_to_pos = defaultdict(set)
            data_dict_tok = [[nlp(sent.replace('bill', 'beak')) for sent in text] for text in self.sent_dict_id]
            for image_idx, image_caps in enumerate(self.sent_dict_id):
                for sent_idx, szsent in enumerate(image_caps):
                    sent = data_dict_tok[image_idx][sent_idx]
                    noun_dict_unfiltered[szsent] = {token.lemma_ for token in sent if token.pos_ == 'NOUN'}  # not in (colors_dict.keys() | colors_inversemap_lower.keys() | {'grey', 'lavender', ('bird' if self.dataset == 'CUB' else 'flower')})])
                    # chunk_heads = [] if 'bird' in get_chunks(sent).values() else ['bird']
                    noun_dict[szsent] = set(
                        [token.lemma_ for token in sent if token.pos_ == 'NOUN' and token.lemma_ != (
                            'bird' if self.dataset == 'CUB' else 'flower')])  # not in (colors_dict.keys() | colors_inversemap_lower.keys() | {'grey', 'lavender', ('bird' if self.dataset == 'CUB' else 'flower')})])

            data_dict_tok.append([nlp(sent.replace('bill', 'beak')) for sent in self.corpus_sents])
            for sent_idx, szsent in enumerate(self.corpus_sents):
                sent = data_dict_tok[-1][sent_idx]
                noun_dict_unfiltered[szsent] = {token.lemma_ for token in sent if token.pos_ == 'NOUN'}  # not in (colors_dict.keys() | colors_inversemap_lower.keys() | {'grey', 'lavender', ('bird' if self.dataset == 'CUB' else 'flower')})])
                # chunk_heads = [] if 'bird' in get_chunks(sent).values() else ['bird']
                noun_dict[szsent] = set(
                    [token.lemma_ for token in sent if token.pos_ == 'NOUN' and token.lemma_ != (
                        'bird' if self.dataset == 'CUB' else 'flower')])  # not in (colors_dict.keys() | colors_inversemap_lower.keys() | {'grey', 'lavender', ('bird' if self.dataset == 'CUB' else 'flower')})])

            def is_neutral(senta, sentb):
                return len(noun_dict[senta].intersection(noun_dict[sentb])) == 0

            # pos_noun_dict = {}
            logging.info("Creating sent2location map")
            for image_idx, image_caps in enumerate(self.sent_dict_id):
                for sent_idx, szsent in enumerate(image_caps):
                    sent_to_pos[szsent].add((image_idx, sent_idx))
            related_imgs = defaultdict(set)
            for vs in sent_to_pos.values():
                for v1, _ in vs:
                    for v2, _ in vs:
                        related_imgs[v1].add(v2)

            logging.info("Creating neg/neut sentences")
            print("Creating neg/neut sentences")
            for ia, senta in enumerate(tqdm(sent_to_pos.keys())):
                for sentb in list(sent_to_pos.keys())[ia:]:
                    pos_senta = next(iter(sent_to_pos[senta]))
                    pos_sentb = next(iter(sent_to_pos[sentb]))
                    if len(sent_to_pos[senta].intersection(sent_to_pos[sentb])) == 0 and pos_sentb[0] not in related_imgs[pos_senta[0]]:
                        if is_neutral(senta, sentb):
                            # if not is_pos(senta, sentb):
                            for tuple_index in sent_to_pos[senta]:
                                self.neut_dict[tuple_index].append(pos_sentb)
                            for tuple_index in sent_to_pos[sentb]:
                                self.neut_dict[tuple_index].append(pos_senta)
                        else:
                            for tuple_index in sent_to_pos[senta]:
                                self.neg_dict[tuple_index].append(pos_sentb)
                            for tuple_index in sent_to_pos[sentb]:
                                self.neg_dict[tuple_index].append(pos_senta)

                for pos_sentb, sentb in enumerate(self.corpus_sents):
                    if is_neutral(senta, sentb):
                        # if not is_pos(senta, sentb):
                        for tuple_index in sent_to_pos[senta]:
                            self.neut_dict_corpus[tuple_index].append(pos_sentb)

            # if 'debug' not in self.split:
            logging.info(f"Saving into cache: {cache_filename}")
            logging.info(f"Sentence count: neg_dict={len(self.neg_dict)}, neut_dict={len(self.neut_dict)} neut_dict_corpus={len(self.neut_dict_corpus)}")
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            pickle.dump([self.neg_dict, self.neut_dict, self.neut_dict_corpus], open(cache_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):

        chance = random.random()
        if chance < 1.0 / 3:
            label = 1
            pos_caps = random.choice(range(len(self.sent_dict_id)))
            pos_caps_a, pos_caps_p = random.sample(range(len(self.sent_dict_id[pos_caps])), k=2)
            sents = [self.sent_dict_id[pos_caps][pos_caps_a], self.sent_dict_id[pos_caps][pos_caps_p]]
            feats = torch.stack([self.precomputed_features[pos_caps][pos_caps_a], self.precomputed_features[pos_caps][pos_caps_p]], dim=0)
            if self.fastdebug and self.log_count < 10:
                self.log_count += 1
                logging.warning(f'Positive\n                      {sents[0]}\n                      {sents[1]}')

        elif chance < 2.0 / 3:
            label = 0
            img_i, cap_i = random.choice(list(self.neg_dict.keys()))
            ni_i, nc_i = random.choice(self.neg_dict[(img_i, cap_i)])
            sents = [self.sent_dict_id[img_i][cap_i], self.sent_dict_id[ni_i][nc_i]]
            feats = torch.stack([self.precomputed_features[img_i][cap_i], self.precomputed_features[ni_i][nc_i]], dim=0)
            if self.fastdebug and self.log_count < 2:
                self.log_count += 1
                logging.warning(f'Negative\n                      {sents[0]}\n                      {sents[1]}')
        else:
            label = 2
            img_i, cap_i = random.choice(list(self.neut_dict.keys()))
            if random.random() > self.p:
                ni_i, nc_i = random.choice(self.neut_dict[(img_i, cap_i)])
                sents = [self.sent_dict_id[img_i][cap_i], self.sent_dict_id[ni_i][nc_i]]
                feats = torch.stack([self.precomputed_features[img_i][cap_i], self.precomputed_features[ni_i][nc_i]], dim=0)
                if self.fastdebug and self.log_count < 2:
                    self.log_count += 1
                    logging.warning(f'Neutral\n                      {sents[0]}\n                      {sents[1]}')
            else:
                n_i = random.choice(self.neut_dict_corpus[(img_i, cap_i)])
                feats = torch.stack([self.precomputed_features[img_i][cap_i], self.precomputed_features_corpus[n_i]], dim=0)
                if self.fastdebug and self.log_count < 2:
                    self.log_count += 1
                    # logging.warning(f'Neutral\n                      {sents[0]}\n                      {sents[1]}')

        return feats, label
