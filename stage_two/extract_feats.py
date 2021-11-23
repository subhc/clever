from sentence_transformers import SentenceTransformer
# sbert_model = 'roberta-large-nli-mean-tokens'
sbert_model = '../checkpoints/stage_one/cub_roberta'
model = SentenceTransformer(sbert_model)
import lmdb
import json
import pickle
import nltk
root = '.'
dataset = 'CUB'
corpus_name = 'aab'
caption_sources = ['captions_prediction_trainval_sup_sat', 'captions_prediction_trainval_sup_aoanet', 'captions_gt']

with torch.no_grad():
    for caption_source in caption_sources:
        print(f'Using captions from /{dataset}/{caption_source}.pickle')
        with open(root + f'/{dataset}/{caption_source}.pickle', 'rb') as handle:
            captions = pickle.load(handle)
        captions_e = {}
        for k, v in tqdm(captions.items()):
            captions_e[k] = model.encode(v, convert_to_tensor=True).cpu()

        with open(f'{dataset}/ours3/{caption_source}_features.pickle', 'wb') as handle:
            pickle.dump(captions_e, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_corpus(class_corpus):
    docs = [(k, cls_text['sents']) for k, cls_text in class_corpus.items()]
    docs_feats = {}
    for k, sents in docs:
        with torch.no_grad():
            feats = model.encode(sents, convert_to_tensor=True).cpu()
        docs_feats[k] = feats
    return docs_feats


import pickle

dataset = 'CUB'
corpus_name = 'aab'
with open(f'..datasets/{dataset}/corpus_{corpus_name}_cleaned.pickle', 'rb') as handle:
    class_corpus = pickle.load(handle)

corpus_features = process_corpus(class_corpus)

with open(f'corpus_{corpus_name}_cleaned_features.pickle', 'wb') as handle:
    pickle.dump(key_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

