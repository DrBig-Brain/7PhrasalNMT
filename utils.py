import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pickle(filepath, data=None):
    import bz2, pickle as pkl
    if data is not None:
        with bz2.BZ2File(filepath, 'wb') as f:
            pkl.dump(data, f)
        return True
    else:
        with bz2.BZ2File(filepath, 'rb') as f:
            return pkl.load(f)

def build_phrase_vocab():
    phrase_types = ['O', 'NP', 'VP', 'PP', 'ADJP', 'ADVP', 'CONJP', 'QP']
    phrase2idx = {p: i for i, p in enumerate(phrase_types)}
    return phrase2idx

def bleu_score(references, candidates):
    from nltk.translate.bleu_score import corpus_bleu
    references = [[sen.split() for sen in refs] for refs in references]
    candidates = [sen.split() for sen in candidates]
    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0)) * 100
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0)) * 100
    bleu3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0)) * 100
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    return bleu1, bleu2, bleu3, bleu4
