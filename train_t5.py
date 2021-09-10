"""
Файнтюнинг ruT5 на датасете, подготовленном в prepare_training_dataset.py
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


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


class NP_Dataset(torch.utils.data.Dataset):
    def __init__(self, device, tokenizer):
        self.device = device
        self.samples = []
        self.max_len = 0
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def append(self, input_text, output_text):
        input_tokens = self.tokenizer(input_text, add_special_tokens=True).input_ids
        output_tokens = self.tokenizer(output_text, add_special_tokens=True).input_ids

        self.max_len = max(self.max_len, len(input_tokens), len(output_tokens))
        self.samples.append((input_tokens, output_tokens))

    def __len__(self):
        return len(self.samples)

    def pad_tokens(self, tokens):
        return tokens + [self.pad_token_id] * (self.max_len - len(tokens))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokens1, tokens2 = self.samples[idx]
        z1 = torch.tensor(self.pad_tokens(tokens1))
        z2 = torch.tensor(self.pad_tokens(tokens2))

        return {'input_ids': z1, 'labels': z2}


def load_samples(dataset_path, device, tokenizer, computed_params):
    train_samples = NP_Dataset(device, tokenizer)
    eval_samples = NP_Dataset(device, tokenizer)

    with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
        for iline, line in enumerate(rdr, start=1):
            fields = line.strip().split('\t')
            input_context = fields[0]
            output_context = fields[1]

            if 0 == (iline % 20):
                eval_samples.append(input_context, output_context)
            else:
                train_samples.append(input_context, output_context)

    computed_params['max_len'] = train_samples.max_len
    return train_samples, eval_samples


def decode_token_ids(tok_ids, tokenizer):
    s = tokenizer.decode(tok_ids).replace('<pad>', ' ')
    if '</s>' in s:
        s = s[:s.index('</s')]
    return s.strip()


def test(model, tokenizer, device, batch_generator):
    model.eval()

    accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(batch_generator, total=len(batch_generator), desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            true_y = batch['labels'].to(device)
            #pred_y = model(input_ids)
            pred_y = model.generate(input_ids=input_ids,
                                    max_length=input_ids.shape[1]+1,
                                    eos_token_id=tokenizer.eos_token_id,
                                    early_stopping=True)
            #pred_y = pred_y[:, 1:]

            for isample in range(input_ids.shape[0]):
                true_str = decode_token_ids(true_y[isample, :], tokenizer)
                pred_str = decode_token_ids(pred_y[isample, 1:], tokenizer)
                acc = jaccard(true_str, pred_str, 3)
                accs.append(acc)

            #a = (true_y == pred_y).sum().item() / float(true_y.shape[0] * true_y.shape[1])
            #accs.append(a)

    acc = np.mean(accs)
    #print('\nTest set: Accuracy: {:.0f}\n'.format(acc))
    return acc


sentinel0 = '<extra_id_0>'
eos_token = '</s>'


if __name__ == '__main__':
    EPOCHS = 10
    tmp_dir = './tmp'
    model_name = 'sberbank-ai/ruT5-base'

    weights_path = os.path.join(tmp_dir, 'rut5.pt')

    tokenizer = T5Tokenizer.from_pretrained(model_name,)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device={}'.format(device))

    # ==== ТРЕНИРОВКА ====
    print('Loading T5 weights...')
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print('Loading dataset...')
    computed_params = dict()
    train_dataset, eval_dataset = load_samples('./data/t5_dataset.txt', device, tokenizer, computed_params)

    batch_size = 10
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Start training...')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    if True:  # fp16
        print('fp16 initialization...')
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optim, opt_level="O1")

    # multi-gpu training (should be after apex fp16 initialization)
    if False:
        print('multi-gpu training initialization...')
        model = torch.nn.DataParallel(model)
    
    
    best_acc = 0.0
    nb_bad_epochs = 0

    for epoch in range(EPOCHS):
        try:
            print('Epoch #{}...'.format(epoch+1))
            for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc='Training'):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                loss = model(input_ids, labels=labels).loss
                loss.backward()
                optim.step()

            acc = test(model, tokenizer, device, eval_dataloader)
            print('')
            if acc > best_acc:
                print('NEW BEST ACC={}'.format(acc))
                best_acc = acc
                nb_bad_epochs = 0
                print('Saving model to "{}"...'.format(weights_path))
                torch.save(model.state_dict(), weights_path)
            else:
                nb_bad_epochs += 1
                print('Score={}, no improvement over current best_acc={}'.format(acc, best_acc))
                if nb_bad_epochs >= 10:
                    print('Early stopping on epoch={} best_acc={}'.format(epoch, best_acc))
                    break
        except KeyboardInterrupt:
            print('Training interrupted on epoch {}.'.format(epoch+1))
            break
