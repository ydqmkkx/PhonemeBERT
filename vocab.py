import string

_special_token = ["[PAD]", "[CLS]", "[SPE]", "[MASK]", "[UNK]"]
_punctuation = list(string.punctuation)+['..']+['...']
_phones = ["AA", "AE", "AH", "AO", "AW", "AY",
    "B",
    "CH",
    "D", "DH",
    "EH", "ER", "EY",
    "F",
    "G",
    "HH",
    "IH", "IY",
    "JH",
    "K",
    "L",
    "M",
    "N", "NG",
    "OW", "OY",
    "P",
    "R",
    "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V",
    "W",
    "Y",
    "Z", "ZH",
]

_token = _special_token + _phones + list(map(lambda s: '##'+s, _phones)) + _punctuation + list(map(lambda s: '##'+s, _punctuation))

_special_range = list(range(len(_special_token)))
_initial_range = list(range(len(_special_token), len(_special_token + _phones))) + list(range(len(_special_token + _phones + list(map(lambda s: '##'+s, _phones))), len(_special_token + _phones + list(map(lambda s: '##'+s, _phones)) + _punctuation)))
_sub_range = list(range(len(_special_token + _phones), len(_special_token + _phones + list(map(lambda s: '##'+s, _phones))))) + list(range(len(_special_token + _phones + list(map(lambda s: '##'+s, _phones)) + _punctuation), len(_token)))

_vocab = {index: value for value, index in enumerate(_token)}
_reversed_vocab = {index: value for index, value in enumerate(_token)}