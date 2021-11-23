import torch
from torch import nn
import torch.nn.functional as F


class SiameseFCN(nn.Module):
    def __init__(self, shared_in_features=1024, shared_out_feature=64, num_labels=2):
        super(SiameseFCN, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, shared_out_feature),
            nn.ReLU())
        self.classifier_in_features = 3 * shared_out_feature
        self.classifier = nn.Linear(self.classifier_in_features, num_labels)

    def forward(self, rep_a, rep_b):
        rep_a = self.shared_encode(rep_a)
        rep_b = self.shared_encode(rep_b)

        return self.get_classifier_scores(rep_a, rep_b)

    def shared_encode(self, x):
        return self.shared_fc(x)

    def get_classifier_scores(self, rep_a, rep_b):
        return self.classifier(torch.cat((rep_a, rep_b, torch.abs(rep_a - rep_b)), 1))
