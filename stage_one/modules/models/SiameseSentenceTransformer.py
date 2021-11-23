import itertools
import json
import logging
import os
from collections import OrderedDict
from typing import Iterable, Dict, List

import torch
from numpy.core.multiarray import ndarray
from torch import nn, Tensor


from modules.models.SentenceTransformerCustom import SentenceTransformerCustom


class SiameseSentenceTransformer(SentenceTransformerCustom):
    def __init__(self,
                 model_name_or_path: str = None,
                 sbert_model: SentenceTransformerCustom = None,
                 sentence_embedding_dimension: int = 1024,
                 num_labels: int = 3,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        if model_name_or_path is not None:
            self.init(model_name_or_path)
        else:
            super(SiameseSentenceTransformer, self).__init__()
            self._modules = OrderedDict()
            self.config_keys = ['sentence_embedding_dimension', 'num_labels', 'concatenation_sent_rep', 'concatenation_sent_difference', 'concatenation_sent_multiplication']
            self.sbert_model = sbert_model
            self.sentence_embedding_dimension = sentence_embedding_dimension
            self.num_labels = num_labels
            self.concatenation_sent_rep = concatenation_sent_rep
            self.concatenation_sent_difference = concatenation_sent_difference
            self.concatenation_sent_multiplication = concatenation_sent_multiplication

            num_vectors_concatenated = 0
            if concatenation_sent_rep:
                num_vectors_concatenated += 2
            if concatenation_sent_difference:
                num_vectors_concatenated += 1
            if concatenation_sent_multiplication:
                num_vectors_concatenated += 1
            logging.info("SiameseSentenceTransformer: #Vectors concatenated: {}".format(num_vectors_concatenated))

            self.in_features = num_vectors_concatenated * self.sentence_embedding_dimension
            self.out_features = num_labels

            self.classifier = nn.Linear(self.in_features, self.out_features)

            self.register_trainable_modules()  # Important
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
            self.device = torch.device(device)
            self.to(device)
            self.classifier.to(device)
            self.print_params()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]]):
        reps = [self.sbert_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        return self.get_classifier_scores(*reps)

    def get_classifier_scores(self, rep_a, rep_b, get_emb=False):
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)

        if get_emb:
            return output, features
        return output

    def encode(self, sentences_list: List[str], batch_size: int = 8, show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding', convert_to_numpy: bool = True) -> List[ndarray]:
        feat_pairs = [self.sbert_model.encode(sentences, batch_size, show_progress_bar, output_value, convert_to_numpy=False) for sentences in sentences_list]
        output = []
        for feat_pair in feat_pairs:
            output.append(self.forward(feat_pair))
        if convert_to_numpy:
            return torch.cat(output, 0).cpu().numpy()
        return torch.cat(output, 0)

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_sentence_input_embedding_dimension(self) -> int:
        return self.in_features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def get_sbert(self):
        return self.sbert_model

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        self.sbert_model.save(os.path.join(output_path, 'encoder'))
        os.makedirs(os.path.join(output_path, 'classifier'), exist_ok=True)
        torch.save(self.classifier.state_dict(), os.path.join(output_path, 'classifier', 'pytorch_model.bin'))

    def init(self, input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        config['sbert_model'] = SentenceTransformerCustom(os.path.join(input_path, 'encoder'))
        self.__init__(**config)
        self.classifier.load_state_dict(torch.load(os.path.join(input_path, 'classifier', 'pytorch_model.bin'), map_location=torch.device('cpu')))

    @staticmethod
    def load(input_path):
        model = SiameseSentenceTransformer()
        model.init(input_path)
        return model

    def print_params(self):
        params = {
            "sentence_embedding_dimension": self.sentence_embedding_dimension,
            "num_labels": self.num_labels,
            "concatenation_sent_rep": self.concatenation_sent_rep,
            "concatenation_sent_difference": self.concatenation_sent_difference,
            "concatenation_sent_multiplication": self.concatenation_sent_multiplication
        }
        logging.info("---------------------------------------")
        logging.info("SiameseSentenceTransformer parameters: ")
        for key, value in params.items():
            logging.info(f"{key:25}:  {value}")
        logging.info("---------------------------------------")
