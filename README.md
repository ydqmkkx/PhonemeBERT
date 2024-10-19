# **PhonemeBERT**
Unofficial PyTorch implementation of Mixed-Phoneme BERT (MP BERT) and Phoneme-Level BERT (PL BERT)

## Introduction
The implementations follow the Hugging Face's frameworks.\
We refer to the implementations of [BERT](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py), [RoBERTa](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta/modeling_deberta.py).

The weights are stored in Hugging Face: [MP BERT](https://huggingface.co/ydqmkkx/mpbert/tree/main), [PL BERT](https://huggingface.co/ydqmkkx/plbert/tree/main).\
More details and pre-training scripts are planned to be open-sourced in the future.

Pre-training configurations:
| | |
| - | - |
| BERT backbone | vanilla BERT<sub>BASE</sub> |
| Corpus | BookCorpus & English Wikipedia |
| Mini-batch   | 2,000 |
| Steps   | 90,000 (about 10 epochs) |
| Max length | 1,024 |
| Optimizer | AdamW |
| Scheduler | Linear |
| Warm-up steps | 9,000 (10%) |
| Peak learning rate | 5 × 10<sup>-4</sup> |
| Environment | 8 × NVIDIA 100 40GB GPUs |
| Training duration | 270 hours (MP BERT), 290 hours (PL BERT)|

## Install
The [PhonemeTokenizer](https://github.com/ydqmkkx/PhonemeTokenizer) is needed:
```bash
pip install git+https://github.com/ydqmkkx/PhonemeTokenizer.git
```
Then the repository should be git-cloned:
```bash
git clone https://github.com/ydqmkkx/PhonemeBERT
cd PhonemeBERT
```
Requirements:
```
torch>=2.0.0
transformers==4.41.2
```

## Usage
```python
from PhonemeTokenizer import PhonemeTokenizer
p_tn = PhonemeTokenizer()
encoding = p_tn("hello, world", return_tensors="pt")

from models import PhonemeBertModel
mpbert = PhonemeBertModel.from_pretrained("ydqmkkx/mpbert")
plbert = PhonemeBertModel.from_pretrained("ydqmkkx/plbert")

# PL BERT only needs phoneme tokens as input
plbert(**encoding)

# MP BERT needs both phoneme and sup-phoneme tokens as input
from utils import sup_phoneme_generator
sup_phoneme_ids = sup_phoneme_generator(encoding.input_ids)
mpbert(**encoding, sup_phoneme_ids=sup_phoneme_ids)

# Our implemented models can generate attention mask automatically
# So we can input without attention_mask
plbert(input_ids=encoding.input_ids)
mpbert(input_ids=encoding.input_ids, sup_phoneme_ids=sup_phoneme_ids)

# The pre-training models can also be loaded
from models import MpbertForPreTraining, PlbertForPreTraining
mpbert_pt = MpbertForPreTraining.from_pretrained("ydqmkkx/mpbert")
plbert_pt = PlbertForPreTraining.from_pretrained("ydqmkkx/plbert")

mpbert_pt(**encoding, sup_phoneme_ids=sup_phoneme_generator(encoding.input_ids))
plbert_pt(**encoding)
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
## Contributors
This project is finished by several members in [Sarulab](https://www.sp.ipc.i.u-tokyo.ac.jp/index-en), the University of Tokyo.
- [Dong Yang](https://ydqmkkx.github.io/)
- [Yuki Saito](https://sython.org/)
- [Takaaki Saeki](https://takaaki-saeki.github.io/)
- [Wataru Nataka](https://wataru-nakata.github.io/)
- [Hiroshi Saruwatari](https://scholar.google.com/citations?user=OS1XAoMAAAAJ&hl=en)
