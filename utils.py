import torch

from tokenizers import Tokenizer, models
sup_phoneme_tokenizer = Tokenizer(models.BPE())
sup_phoneme_tokenizer = sup_phoneme_tokenizer.from_file('SupPhonemeTokenizer')

import json
with open('token2phoneme.json', 'r') as f:
    token2phoneme = json.load(f)

def arpabet2ipa(ps):
    ipas = []
    for idx in ps:
        ipa = token2phoneme[str(idx)]
        if ipa['initial']:
            ipas.append(ipa['ipa'])
        else:
            ipas[-1] += ipa['ipa']
    return ipas

def sup_phoneme_generator_train(ps):
    ps = arpabet2ipa(ps)
    ids = []
    tokens = []
    lengths = []
    for p in ps:
        encoding = sup_phoneme_tokenizer.encode(p)
        ids += encoding.ids
        tokens += encoding.tokens
        for token in encoding.tokens:
            if token in ["[PAD]", "[CLS]", "[SPE]", "[MASK]", "[UNK]"]:
                lengths += [1]
            else:
                lengths += [len(token)]
    cum_lengths = np.cumsum(lengths).tolist()
    upsampling_matrix = np.zeros((len(lengths), cum_lengths[-1]), int)
    downsampling_matrix = np.zeros((len(lengths), cum_lengths[-1]), float)
    idx = 0
    for l1, l2 in zip([0]+cum_lengths[:-1], cum_lengths):
        upsampling_matrix[idx][l1:l2] = 1
        downsampling_matrix[idx][l1:l2] = 1/lengths[idx]
        idx += 1
    return np.array(ids)@upsampling_matrix, downsampling_matrix, len(lengths)

def sup_phoneme_generator_infer(ps):
    sup_ids = []
    ps = arpabet2ipa(ps)
    for p in ps:
        encoding = sup_phoneme_tokenizer.encode(p)
        ids = encoding.ids
        tokens = encoding.tokens
        for idx, token in zip(ids, tokens):
            if token in ["[PAD]", "[CLS]", "[SPE]", "[MASK]", "[UNK]"]:
                sup_ids += [idx]
            else:
                sup_ids += [idx] * len(token)
    return sup_ids

def sup_phoneme_generator(phoneme_ids):
    sp_ids = phoneme_ids.tolist()
    sp_ids = [torch.tensor(sup_phoneme_generator_infer(sp_ids[i])) for i in range(len(sp_ids))]
    sp_ids = torch.stack(sp_ids).to(phoneme_ids.device)
    return sp_ids