# **PhonemeBERT**
Unofficial PyTorch implementation of Mixed-Phoneme BERT (MP BERT) and Phoneme-Level BERT (PL BERT)

## Introduction
The implementations follow the Hugging Face's frameworks.\
We refer to the implementations of [BERT](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py), [RoBERTa](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta/modeling_deberta.py).\
The pre-training scripts are planned to be open-sourced in the future.

Pre-training configurations:
| | |
| - | - |
| BERT backbone | vanilla BERT<sub>BASE</sub>
| Corpus | BookCorpus & English Wikipedia |
| Mini-batch   | 2,000 |
| Steps   | 90,000 (about 10 epochs) |
| Max length | 1,024 |
| Optimizer | AdamW |
| Scheduler | Linear |
| Warm-up steps | 9,000 (10%) |
| Peak learning rate | 5 × 10<sup>-4</sup> |

## Install
The [PhonemeTokenizer](https://github.com/ydqmkkx/PhonemeTokenizer) is needed:
```bash
pip install git+https://github.com/ydqmkkx/PhonemeTokenizer.git
```
Requirements:
```
torch>=2.0.0
transformers==4.41.2
```
Then the repository should be git-cloned:
```bash
git clone https://github.com/ydqmkkx/PhonemeBERT
cd PhonemeBERT
```

## Usage
```python
from PhonemeTokenizer import PhonemeTokenizer
from models import PhonemeBertModel

p_tn = PhonemeTokenizer()
mpbert = PhonemeBertModel.from_pretrained("ydqmkkx/mpbert")
plbert = PhonemeBertModel.from_pretrained("ydqmkkx/plbert")

encoding = p_tn("hello, world", return_tensors="pt")

# PLBERT only needs phoneme tokens as input
plbert(**encoding)

# MPBERT needs both phoneme and sup-phoneme tokens as input
from utils import sup_phoneme_generator

sup_phoneme_ids = sup_phoneme_generator(encoding.input_ids)
mpbert(**encoding, sup_phoneme_ids=sup_phoneme_ids)

# Our implemented models can generate attention mask automatically
# So we can input without attention_mask
plbert(input_ids=encoding.input_ids)
mpbert(input_ids=encoding.input_ids, sup_phoneme_ids=sup_phoneme_ids)
```

## Citation
```bibtex
@CONFERENCE{zhang22mpbert,
  title = {Mixed-Phoneme {BERT}: Improving {BERT} with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech},
  author = {G. Zhang and K. Song and X. Tan and D. Tan and Y. Yan and Y. Liu and G. Wang and W. Zhou and T. Qin and T. Lee and S. Zhao},
  booktitle = {Proc. Interspeech},
  pages={456--460},
  year = {2022},
}

@CONFERENCE{li23plbert,
  title = {Phoneme-Level {BERT} for Enhanced Prosody of Text-to-Speech with Grapheme Predictions},
  author = {Y. A. Li and C. Han and X. Jiang and N. Mesgarani},
  booktitle = {Proc. ICASSP},
  year = {2023},
}
```