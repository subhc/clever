import json
import pickle

import numpy as np
import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, root, model, dataset, feature_model, corpus_name, loss_mult, loss_type="normal", three=False):
        super(CrossEntropyLoss, self).__init__()
        self.model = model
        self.dataset = dataset
        self.corpus_name = corpus_name
        self.three = three
        self.loss_mult = loss_mult
        self.loss_type = loss_type
        with open(root + f'/{self.dataset}/image_data.json', 'r') as handle:
            self.image_data = json.load(handle)

        self.classes = sorted(list({v['class_name'] for v in self.image_data['images']}))
        self.present_classes = list(range(len(self.classes)))

        with open(root + f'/{self.dataset}/{feature_model}/corpus_{self.corpus_name}_features.pickle', 'rb') as handle:
            self.class_corpus = pickle.load(handle)

        self.document_list = [self.class_corpus[cls] for cls in self.classes]
        document_lengths = [len(docs) for docs in self.document_list]
        document_lengths = [0] + np.cumsum(document_lengths, 0).tolist()
        self.document_spans = [(document_lengths[i], document_lengths[i + 1]) for i in range(len(document_lengths) - 1)]
        assert all(docs.size(0) == end - start for docs, (start, end) in zip(self.document_list, self.document_spans))
        self.document_list = torch.cat(self.document_list, dim=0)
        self.criterion_sm = nn.CrossEntropyLoss() if self.three else nn.BCEWithLogitsLoss()

        if self.loss_type == 'normal':
            self.reduce_docs = self.reduce_docs_normal
        elif self.loss_type == 'doublesoftmax':
            self.reduce_docs = self.reduce_docs_doublesoftmax
        elif self.loss_type == 'max':
            self.reduce_docs = self.reduce_docs_max

    def criterion(self, x):
        dots = torch.matmul(x, x.t())  # batch x batch
        return (dots - 2 * torch.diag(dots.diag())).mean()

    def forward(self, x, labels=None, return_scores=False):
        x = self.model.shared_encode(x)  # batch x num_captions x siamese_dim
        if return_scores:
            return self.scores_on_documents_flattened(x)  # batch x num_docs
        scores = self.model.get_classifier_scores(x[:, 0], x[:, 1]).squeeze(1)
        loss = self.criterion_sm(scores, labels if self.three else labels.float())
        x = self.scores_on_documents_flattened(x)
        x = self.criterion(x)
        loss += x*self.loss_mult
        return loss

    def scores_on_documents_flattened(self, sentence_embs):
        batch, num_captions, _ = sentence_embs.size()
        sentence_embs = sentence_embs.view(-1, sentence_embs.size(2))
        document_embeddings = self.model.shared_encode(self.document_list)  # num_docs*num_sents x siamese_dim
        doc_scores = self.classifier_on_embeddings_product(sentence_embs, document_embeddings)  # batch*num_captions x num_docs*num_sents
        doc_scores = [doc_scores[:, start:end].view(batch, num_captions, end - start, doc_scores.size(2)) for start, end in self.document_spans]  # [ batch x num_caps x num_sents ] x num_docs
        if self.three:
            doc_scores = [doc.softmax(dim=3) for doc in doc_scores]  # [ batch  x num_caps x num_sents x 3 ] x num_docs
            doc_scores = [(doc[:, :, :, 1] - doc[:, :, :, 0]) for doc in doc_scores]  # [ batch x num_caps x num_sents ] x num_docs
        else:
            doc_scores = [doc[:, :, :, 0] for doc in doc_scores]  # [ batch x num_caps x num_sents ] x num_docs

        doc_scores = self.reduce_docs(doc_scores)  # [ batch x num_caps ] x num_docs
        doc_scores = [doc.mean(dim=1) for doc in doc_scores]  # [ batch x num_caps x num_sents ] x num_docs
        doc_scores = torch.cat(doc_scores, 1)  # batch x num_docs
        doc_scores = doc_scores.softmax(1)
        return doc_scores

    def reduce_docs_normal(self, doc_scores):   # [ batch x num_caps x num_sents ] x docs
        return [doc.mean(dim=2, keepdim=True) for doc in doc_scores]  # [ batch x num_caps x 1 ] x docs

    def reduce_docs_max(self, doc_scores):   # [ batch x num_caps x num_sents ] x docs
        return [doc.max(dim=2, keepdim=True)[0] for doc in doc_scores]  # [ batch x num_caps x 1 ] x docs

    def reduce_docs_doublesoftmax(self, doc_scores):   # [ batch x num_caps x num_sents ] x docs
        return [(doc * doc.softmax(dim=2)).sum(dim=2, keepdim=True) for doc in doc_scores]  # [ batch x num_caps x 1 ] x docs

    def classifier_on_embeddings_product(self, feat_a, feat_b):
        output = self.model.get_classifier_scores(torch.repeat_interleave(feat_a, repeats=feat_b.size(0), dim=0), feat_b.repeat(feat_a.size(0), 1))
        return output.view(len(feat_a), len(feat_b), -1)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.document_list = self.document_list.to(*args, **kwargs)
        return self
