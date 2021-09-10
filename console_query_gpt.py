"""
Консольная интерактивная проверка модели рандомизатора, натренированной на датасете
gpt_dataset.txt (см. prepare_training_dataset.py)
"""

import os
import io
import logging.handlers
import random

import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import terminaltables

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError

import extractors


class RugptGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self, model_name_or_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def generate_output(self, context, num_return_sequences=10):
        temperature = 0.9
        beam_k = 10
        beam_p = 0.9
        repetition_penalty = 1.0
        prompt_text = context + ' #'
        stop_token = "</s>"
        length = 100

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=beam_k,
            top_p=beam_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.encoder['<pad>'],
            eol_token_id=self.tokenizer.encoder['|']
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("ruGPT2Large:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

            if '#' in total_sequence:
                total_sequence = total_sequence[: total_sequence.find('#')]

            total_sequence = total_sequence.strip()
            generated_sequences.add(total_sequence)
            #print(total_sequence)

        return list(generated_sequences)


def compose_constituent(root_token, tokens, token2children):
    extracted_tokens = [root_token]
    extract_child_tokens(root_token, tokens, token2children, extracted_tokens)
    extracted_tokens = sorted(extracted_tokens, key=lambda t: int(t.id))

    # Убираем частицу в начале NP или PP:
    # проявляется и отношение к ним собирательницы
    #            ^^^
    if root_token.upos in ('NOUN', 'PROPN', 'PRON') and extracted_tokens[0].upos == 'PART' and extracted_tokens[0].form.lower() != 'не':
        extracted_tokens = extracted_tokens[1:]

    if extracted_tokens[-1].form in '.!?,:-':
        extracted_tokens = extracted_tokens[:-1]

    return extracted_tokens


def extract_child_tokens(root_token, tokens, token2children, extracted_tokens):
    for child in token2children.get(root_token.id, []):
        extracted_tokens.append(child)
        extract_child_tokens(child, tokens, token2children, extracted_tokens)


def extract_nps(parsing):
    token2children = dict()
    for token in parsing:
        if token.head != '0':
            if token.head in token2children:
                token2children[token.head].append(token)
            else:
                token2children[token.head] = [token]

    # Ищем глагольное сказуемое
    root = None
    for t in parsing:
        if t.deprel == 'root' and t.upos == 'VERB':
            root = t
            break

    if root:
        # Ищем существительные, присоединенные прямо к сказуемому
        for t in parsing:
            if t.upos in ('NOUN',) and t.head == root.id:
                # Исключаем PP, для этого проверим, что у существительного нет прикрепленного предлога.
                if not any((t2.upos == 'ADP' and t2.head == t.id) for t2 in parsing):
                    np_tokens = compose_constituent(t, parsing, token2children)
                    yield np_tokens

        # Ищем существительные, входящие в PP
        for t in parsing:
            if t.upos in ('NOUN',):
                for t2 in token2children.get(t.id, []):
                    if t2.upos == 'ADP':
                        np_tokens = compose_constituent(t, parsing, token2children)
                        if np_tokens[0].id == t2.id: # исключаем случаи PP с префиксальной частицей "[и на старуху] бывает проруха"
                            # Уберем собственно предлог из цепочки токенов
                            np_tokens = [t3 for t3 in np_tokens if t3.id != t2.id]

                            yield np_tokens


if __name__ == '__main__':
    generator = RugptGenerator()
    generator.load('./tmp/rugpt_model')

    print('Loading models...')
    model = Model.load('/home/inkoziev/polygon/GramEval2020/tmp/udpipe_syntagrus.model')
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    udp_error = ProcessingError()

    while True:
        input_text = input(':> ').strip()
        if len(input_text) == 0:
            break

        processed = pipeline.process(input_text, udp_error)
        parsing = pyconll.load_from_string(processed)[0]

        for c_type, c_tokens in extractors.extract_constituents(parsing):
            c_ids = [t.id for t in c_tokens]
            input_tokens = []
            for t in parsing:
                if t.id == c_tokens[0].id:
                    # Первый токен в составляющей
                    input_tokens.append('[{}]'.format(c_type))
                elif t.id in c_ids:
                    # Второй и последующие токены в составляющей пропускаем
                    pass
                else:
                    input_tokens.append(t.form)

            input_text = ' '.join(input_tokens)

            print('Input: {}'.format(input_text))
            outputs = generator.generate_output(input_text, num_return_sequences=5)
            table = [[c_type, 'result']]
            for i, output in enumerate(outputs, start=1):
                if output:
                    text2 = input_text.replace('[{}]'.format(c_type), output)
                    table.append([output, text2])

            table = terminaltables.AsciiTable(table)
            print(table.table)
            print('')
