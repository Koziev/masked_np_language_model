"""
Интерактивная проверка работы отфайнтюненной ruT5 (см. train_t5.py)
"""

import io
import re
import os

import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader

import numpy as np


sentinel0 = '<extra_id_0>'
eos_token = '</s>'


if __name__ == '__main__':
    EPOCHS = 1
    tmp_dir = './tmp'
    model_name = 'sberbank-ai/ruT5-large'

    weights_path = os.path.join(tmp_dir, 'rut5_for_np_substituton.pt')

    tokenizer = T5Tokenizer.from_pretrained(model_name,)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device={}'.format(device))

    t5config = T5Config.from_pretrained(model_name, )
    model = T5ForConditionalGeneration(t5config)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    t5_input = 'Голодная кошка ловит <extra_id_0> .'
    input_ids = tokenizer(t5_input, return_tensors='pt').input_ids
    out_ids = model.generate(input_ids=input_ids,
                             max_length=40,
                             eos_token_id=tokenizer.eos_token_id,
                             early_stopping=True)

    t5_output = tokenizer.decode(out_ids[0][1:])
    print('t5_output={}'.format(t5_output))
