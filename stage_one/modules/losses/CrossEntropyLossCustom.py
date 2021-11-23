import logging

import torch
from torch import nn, Tensor

class CrossEntropyLossCustom(nn.Module):
    def __init__(self, model, ce_weight=None):
        super(CrossEntropyLossCustom, self).__init__()
        self.model = model
        self.ce_weight = ce_weight
        if self.ce_weight is None:
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.ce_weight).cuda())

    def forward(self, sentence_features, labels: Tensor):
        reps = [self.model.sbert_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        cls_score = self.model.get_classifier_scores(*reps)

        return self.loss_fct(cls_score, labels.view(-1))

    def print_params(self):
        params = {
            "ce_weight": self.ce_weight
        }
        logging.info("---------------------------------------")
        logging.info("CrossEntropyLoss parameters: ")
        for key, value in params.items():
            logging.info(f"{key:25}:  {value}")
        logging.info("---------------------------------------")
