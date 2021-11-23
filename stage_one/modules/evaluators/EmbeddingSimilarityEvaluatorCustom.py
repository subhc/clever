import csv
import logging
import os

import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.models import SentenceTransformerCustom

paired_cosine_distances = TripletDistanceMetric.COSINE
paired_euclidean_distances = TripletDistanceMetric.EUCLIDEAN
paired_manhattan_distances = TripletDistanceMetric.MANHATTAN


# https://github.com/pytorch/pytorch/issues/21987
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def pairwise_dot(x, y):
    b = x.size(0)
    s = x.size(1)
    return torch.bmm(x.view(b, 1, s), y.view(b, s, 1)).reshape(-1)


example_type_map = [[('negative', 'chunksent'), ('negative', 'wholesent'), ('negative', 'aab_chunkimg')], [('positive', 'chunksent'), ('positive', 'wholesent'), ('positive', 'aab_chunkimg')], [('neutral', 'chunksent'), ('neutral', 'wholesent'), ('neutral', 'aab_chunkimg'), ('neutral', 'randneutral')]]


class EmbeddingSimilarityEvaluatorCustom(EmbeddingSimilarityEvaluator):
    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None, writer=None, prefix='train', noun_vocab=None):
        super(EmbeddingSimilarityEvaluatorCustom, self).__init__(dataloader, main_similarity, name, show_progress_bar)
        self.writer = writer
        self.noun_vocab = noun_vocab
        self.prefix = prefix

    def __call__(self, model: SentenceTransformerCustom, output_path: str = None, epoch: int = -1, steps: int = -1,  num_batches=1) -> float:
        model.eval()
        sbert_model = model.get_sbert()
        embeddings1 = []
        embeddings2 = []
        example_types_tensor = []
        labels = []
        labels_tensor = []
        prediction_list = []
        acc = []
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation the model on " + self.prefix + " dataset" + out_txt)

        self.dataloader.collate_fn = sbert_model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc=f"Evaluating {self.prefix} NLI set")
        for step, batch in enumerate(iterator):
            features, label_ids, example_types, nouns, counts = model.batch_to_device(batch, self.device)
            with torch.no_grad():
                embs = [sbert_model(sent_features)['sentence_embedding'] for sent_features in features]
                emb1, emb2 = embs[0], embs[1]
                output = model.get_classifier_scores(emb1, emb2, get_emb=False)
            pred_out = output.argmax(1)
            acc.extend(pred_out == label_ids)
            labels.extend(label_ids.to("cpu").numpy())
            labels_tensor.extend(label_ids)
            prediction_list.extend(pred_out)
            example_types_tensor.extend(example_types)
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)

        embeddings1 = torch.stack(embeddings1, 0)
        embeddings2 = torch.stack(embeddings2, 0)
        acc = torch.stack(acc, 0).float()
        labels_tensor = torch.stack(labels_tensor, 0)

        try:
            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            cosine_scores = cosine_scores.cpu().numpy()
        except Exception as e:
            print(embeddings1)
            print(embeddings2)
            raise e

        dot_products = pairwise_dot(embeddings1, embeddings2).cpu().numpy()

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logging.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logging.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))
        logging.info("Accuracy:\tPositive: {:.2f}\tNegative: {:.2f}\tNeutral: {:.2f}\tOverall: {:.2f}".format(
            torch.mean(acc[labels_tensor == 1]).item() * 100, torch.mean(acc[labels_tensor == 0]).item() * 100, torch.mean(acc[labels_tensor == 2]).item() * 100, torch.mean(acc).item() * 100))
        if self.writer is not None:
            if epoch == -1:
                n = 0
            else:
                n = (epoch + 1) if steps == -1 else epoch + steps / num_batches
                n = int(n * 1000)  # Tensorboard does not support fractional steps

            # self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/pearson_cosine', eval_pearson_cosine, n)
            # self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/eval_spearman_cosine', eval_spearman_cosine, n)
            # self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/eval_pearson_dot', eval_pearson_dot, n)
            # self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/eval_spearman_dot', eval_spearman_dot, n)
            self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/accuracy_negative', torch.mean(acc[labels_tensor == 0]).item(), n)
            self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/accuracy_positive', torch.mean(acc[labels_tensor == 1]).item(), n)
            self.writer.add_scalar('evaluation_sbert/' + self.prefix + '/accuracy_neutral', torch.mean(acc[labels_tensor == 2]).item(), n)
            # for label_idx in range(3):
            #     for type_idx in range(len(example_type_map[label_idx])):
            #         label_name, type_name = example_type_map[label_idx][type_idx]
            #         self.writer.add_scalar(f'sbert_evaluation/{self.prefix}/acc_{label_name}/{type_name}', torch.mean(acc[torch.logical_and(labels_tensor == label_idx, example_types_tensor == type_idx)]).item(), n)

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_dot, eval_spearman_dot])

        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")
